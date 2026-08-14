"""Minimal RealBasicVSR inference architecture adapted from MMagic v1.2.0.

The upstream implementation is Apache-2.0 licensed:
https://github.com/open-mmlab/mmagic/tree/c749dcc7172d198ac2a27c3e5a4d2181640f0fd5/mmagic/models/editors
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


def flow_warp(
    tensor: torch.Tensor,
    flow: torch.Tensor,
    *,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp a feature tensor with pixel-unit optical flow."""

    if tensor.shape[-2:] != flow.shape[1:3]:
        raise ValueError("tensor and flow spatial dimensions must match")
    _, _, height, width = tensor.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=tensor.dtype),
        torch.arange(width, device=flow.device, dtype=tensor.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[..., 0] / max(width - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[..., 1] / max(height - 1, 1) - 1.0
    normalized = torch.stack((grid_flow_x, grid_flow_y), dim=3).to(tensor.dtype)
    return functional.grid_sample(
        tensor,
        normalized,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


class ResidualBlockNoBN(nn.Module):
    """Two-convolution residual block with MMagic-compatible state keys."""

    def __init__(self, mid_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.conv2(self.relu(self.conv1(tensor)))


class ResidualBlocksWithInputConv(nn.Module):
    """Input projection followed by the official residual block stack."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*(ResidualBlockNoBN(out_channels) for _ in range(num_blocks))),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.main(tensor)


class PixelShufflePack(nn.Module):
    """Convolution plus pixel shuffle with MMagic-compatible state keys."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * 4,
            3,
            padding=1,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return functional.pixel_shuffle(self.upsample_conv(tensor), 2)


class ConvRelu(nn.Module):
    """Subset of mmcv ConvModule required by SPyNet."""

    def __init__(self, in_channels: int, out_channels: int, *, activate: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias=True)
        self._activate = activate

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv(tensor)
        return functional.relu(tensor, inplace=True) if self._activate else tensor


class SPyNetBasicModule(nn.Module):
    """One optical-flow pyramid refinement module."""

    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvRelu(8, 32, activate=True),
            ConvRelu(32, 64, activate=True),
            ConvRelu(64, 32, activate=True),
            ConvRelu(32, 16, activate=True),
            ConvRelu(16, 2, activate=False),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.basic_module(tensor)


class SPyNet(nn.Module):
    """Six-level spatial-pyramid optical flow estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _compute_flow(self, reference: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = reference.shape
        references = [(reference - self.mean) / self.std]
        supports = [(support - self.mean) / self.std]
        for _ in range(5):
            references.append(
                functional.avg_pool2d(references[-1], 2, 2, count_include_pad=False)
            )
            supports.append(
                functional.avg_pool2d(supports[-1], 2, 2, count_include_pad=False)
            )
        references.reverse()
        supports.reverse()
        flow = reference.new_zeros(batch, 2, height // 32, width // 32)
        for level, module in enumerate(self.basic_module):
            flow_up = (
                flow
                if level == 0
                else functional.interpolate(
                    flow,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                * 2.0
            )
            warped = flow_warp(
                supports[level],
                flow_up.permute(0, 2, 3, 1),
                padding_mode="border",
            )
            flow = flow_up + module(
                torch.cat((references[level], warped, flow_up), dim=1)
            )
        return flow

    def forward(self, reference: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        height, width = reference.shape[-2:]
        padded_height = ((height + 31) // 32) * 32
        padded_width = ((width + 31) // 32) * 32
        resized_reference = functional.interpolate(
            reference,
            size=(padded_height, padded_width),
            mode="bilinear",
            align_corners=False,
        )
        resized_support = functional.interpolate(
            support,
            size=(padded_height, padded_width),
            mode="bilinear",
            align_corners=False,
        )
        flow = functional.interpolate(
            self._compute_flow(resized_reference, resized_support),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        flow[:, 0] *= width / padded_width
        flow[:, 1] *= height / padded_height
        return flow


class BasicVSRNet(nn.Module):
    """Bidirectional BasicVSR propagation and native x4 reconstruction."""

    def __init__(self, mid_channels: int = 64, num_blocks: int = 20) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3,
            mid_channels,
            num_blocks,
        )
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3,
            mid_channels,
            num_blocks,
        )
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels)
        self.upsample2 = PixelShufflePack(mid_channels, 64)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def _compute_flow(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, channels, height, width = frames.shape
        first = frames[:, :-1].reshape(-1, channels, height, width)
        second = frames[:, 1:].reshape(-1, channels, height, width)
        backward = self.spynet(first, second).view(batch, time - 1, 2, height, width)
        forward = self.spynet(second, first).view(batch, time - 1, 2, height, width)
        return forward, backward

    def forward(self, frames: torch.Tensor, *, output_scale: int) -> torch.Tensor:
        batch, time, _, height, width = frames.shape
        flows_forward, flows_backward = self._compute_flow(frames)
        backward_features: list[torch.Tensor] = []
        propagated = frames.new_zeros(batch, self.mid_channels, height, width)
        for index in range(time - 1, -1, -1):
            if index < time - 1:
                propagated = flow_warp(
                    propagated,
                    flows_backward[:, index].permute(0, 2, 3, 1),
                )
            propagated = self.backward_resblocks(
                torch.cat((frames[:, index], propagated), dim=1)
            )
            backward_features.append(propagated)
        backward_features.reverse()

        outputs: list[torch.Tensor] = []
        propagated = torch.zeros_like(propagated)
        for index in range(time):
            current = frames[:, index]
            if index > 0:
                propagated = flow_warp(
                    propagated,
                    flows_forward[:, index - 1].permute(0, 2, 3, 1),
                )
            propagated = self.forward_resblocks(
                torch.cat((current, propagated), dim=1)
            )
            output = self.lrelu(
                self.fusion(torch.cat((backward_features[index], propagated), dim=1))
            )
            output = self.lrelu(self.upsample1(output))
            output = self.lrelu(self.upsample2(output))
            output = self.lrelu(self.conv_hr(output))
            output = self.conv_last(output) + self.img_upsample(current)
            if output_scale == 2:
                output = functional.interpolate(
                    output,
                    scale_factor=0.5,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            outputs.append(output)
        return torch.stack(outputs, dim=1)


class RealBasicVSRNet(nn.Module):
    """Sequential-cleaning RealBasicVSR generator used by the official model."""

    def __init__(self) -> None:
        super().__init__()
        self.dynamic_refine_thres = 1.5 / 255.0
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, 64, 20),
            nn.Conv2d(64, 3, 3, 1, 1, bias=True),
        )
        self.basicvsr = BasicVSRNet(64, 20)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, frames: torch.Tensor, *, output_scale: int) -> torch.Tensor:
        _, time, _, _, _ = frames.shape
        cleaned = frames.clone()
        for _ in range(3):
            residues = []
            for index in range(time):
                residue = self.image_cleaning(cleaned[:, index])
                cleaned[:, index] = cleaned[:, index] + residue
                residues.append(residue)
            if torch.mean(torch.abs(torch.stack(residues, dim=1))) < self.dynamic_refine_thres:
                break
        return self.basicvsr(cleaned, output_scale=output_scale)
