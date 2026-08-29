#!/usr/bin/env bash
# nvchk/build_probes.sh — compile every probe in this directory.
#
# Run `./fetch_headers.sh` first; the probes `#include "ffnvcodec/..."`.
#
# Plain gcc on purpose.  These deliberately do NOT go through Cargo or link the
# repo's import stub: they load `nvcuda.dll` / `nvEncodeAPI64.dll` with
# `LoadLibraryA` + `GetProcAddress`, so nothing about build.rs, `build/cuda.def`,
# or the `/DELAYLOAD` flags can influence the result.  That independence is the
# point — a probe that shared the build system could reproduce a build-system bug
# and be mistaken for a driver measurement.
#
# Usage:  ./build_probes.sh          (from this directory)

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ffnvcodec/nvEncodeAPI_n12.2.72.0.h ]; then
    echo "ffnvcodec/ headers missing — run ./fetch_headers.sh first" >&2
    exit 1
fi

CC=${CC:-gcc}
FLAGS="-O1 -I."

build() {
    local out=$1; shift
    echo "== $out"
    # shellcheck disable=SC2086
    $CC $FLAGS -o "$out" "$@"
}

# Layout/enumerant probes: one binary per API version, selected by -DNVHDR.
# All three must agree for a constant to be described as version-stable.
for tag in n12.0.16.0 n12.2.72.0 n13.0.19.0; do
    build "probe_$tag.exe" probe.c -DNVHDR="\"ffnvcodec/nvEncodeAPI_$tag.h\""
done

# Behavioural probes.  These TOUCH THE DRIVER and encode real frames; two of
# them can kill the process by design (see the pitch=0 rung), which is why each
# takes its rung on argv and is meant to be run one process at a time.
build nv12_probe.exe       nv12_probe.c
build nv12_pitch_probe.exe nv12_pitch_probe.c
build extbuf_probe.exe     extbuf_probe.c    -ld3d12
build d3d12_buf_probe.exe  d3d12_buf_probe.c -ld3d12 -ldxgi

echo
echo "built:"
ls -1 ./*.exe
