#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
runtime_root="$repo_root/.runtime"
download_root="$runtime_root/downloads"
tool_root="$runtime_root/tools/ffmpeg-n8.1.2-34-g9b6c8969e0"
archive_name="ffmpeg-n8.1.2-34-g9b6c8969e0-linux64-gpl-8.1.tar.xz"
archive_path="$download_root/$archive_name"
archive_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-13-17-03/$archive_name"
archive_sha256="5ca9b490fc4385de7545979e6320608bfbb769242a96e8eb3d45e103082b26d0"
archive_size_bytes="125776800"

mkdir -p "$download_root" "$runtime_root/tools"

if [[ ! -f "$archive_path" ]]; then
    archive_partial="$archive_path.partial"
    if [[ -e "$archive_partial" ]]; then
        echo "stale FFmpeg partial download exists: $archive_partial" >&2
        exit 1
    fi
    curl --location --fail --retry 3 --output "$archive_partial" "$archive_url"
    mv -- "$archive_partial" "$archive_path"
fi

actual_size="$(stat -c '%s' "$archive_path")"
if [[ "$actual_size" != "$archive_size_bytes" ]]; then
    echo "FFmpeg archive size mismatch: expected=$archive_size_bytes actual=$actual_size" >&2
    exit 1
fi
actual_sha256="$(sha256sum "$archive_path" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$archive_sha256" ]]; then
    echo "FFmpeg archive SHA-256 mismatch: expected=$archive_sha256 actual=$actual_sha256" >&2
    exit 1
fi

if [[ ! -x "$tool_root/bin/ffmpeg" || ! -x "$tool_root/bin/ffprobe" ]]; then
    if [[ -e "$tool_root" ]]; then
        echo "FFmpeg tool directory exists but is incomplete: $tool_root" >&2
        exit 1
    fi
    tool_partial="$tool_root.partial"
    if [[ -e "$tool_partial" ]]; then
        echo "stale FFmpeg partial extraction exists: $tool_partial" >&2
        exit 1
    fi
    mkdir "$tool_partial"
    tar -xJf "$archive_path" --strip-components=1 -C "$tool_partial"
    if [[ ! -x "$tool_partial/bin/ffmpeg" || ! -x "$tool_partial/bin/ffprobe" ]]; then
        echo "verified FFmpeg archive does not contain executable binaries" >&2
        exit 1
    fi
    mv -- "$tool_partial" "$tool_root"
fi

ffmpeg_version="$($tool_root/bin/ffmpeg -version | sed -n '1p')"
ffprobe_version="$($tool_root/bin/ffprobe -version | sed -n '1p')"
if [[ "$ffmpeg_version" != *"n8.1.2-34-g9b6c8969e0"* ]]; then
    echo "unexpected FFmpeg version: $ffmpeg_version" >&2
    exit 1
fi
if [[ "$ffprobe_version" != *"n8.1.2-34-g9b6c8969e0"* ]]; then
    echo "unexpected ffprobe version: $ffprobe_version" >&2
    exit 1
fi

printf '%s\n' "$ffmpeg_version" "$ffprobe_version"
