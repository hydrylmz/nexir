#!/usr/bin/env bash
# nvchk/fetch_headers.sh — fetch the vendor headers the probes compile against.
#
# The headers are NOT vendored: they are ~900 KB of third-party source, they are
# not Nexir's to redistribute, and pinning them by TAG is a stronger provenance
# claim than a copy in our tree (a copy can be edited; a tag cannot).  This
# script materialises `ffnvcodec/` beside the probes, which is where every
# `#include "ffnvcodec/..."` in them looks.
#
# The three tags are the ones the measurements in
# `src/interop/ffi/nvenc.rs` were cross-checked across: every enumerant the FFI
# hard-codes agrees in all three, which is why those constants are described as
# stable rather than as "true on this driver".  n12.2.72.0 is the one matching
# the driver the behavioural rungs ran on (NVENC API 12.2).
#
# Usage:  ./fetch_headers.sh          (from this directory)

set -euo pipefail
cd "$(dirname "$0")"

REPO=https://github.com/FFmpeg/nv-codec-headers.git

# tag -> the commit it resolved to when these measurements were taken.  Recorded
# so a re-run that silently gets different content is visible rather than
# assumed: a moved tag would change what "verified against the header" means.
declare -A EXPECTED=(
  [n12.0.16.0]=c5e4af74850a616c42d39ed45b9b8568b71bf8bf
  [n12.2.72.0]=c69278340ab1d5559c7d7bf0edf615dc33ddbba7
  [n13.0.19.0]=e844e5b26f46bb77479f063029595293aa8f812d
)

mkdir -p ffnvcodec

# Clone into a RELATIVE subdirectory rather than $(mktemp -d).  On Windows/MSYS,
# `mktemp -d` returns an MSYS path like `/tmp/tmp.XXXX`, and `git` is a NATIVE
# program that resolves `/tmp` against the current drive (C:\tmp) while MSYS `cp`
# resolves it against the MSYS mount root (C:\Program Files\Git\tmp).  The clone
# then succeeds and the copy fails with "cannot stat" on a file that plainly
# exists.  A relative path is resolved identically by both.
work=.hdrwork
rm -rf "$work"
trap 'rm -rf "$work"' EXIT

for tag in "${!EXPECTED[@]}"; do
    echo "== $tag"
    git clone --quiet --depth 1 -b "$tag" "$REPO" "$work/$tag"
    got=$(git -C "$work/$tag" rev-parse HEAD)
    if [ "$got" != "${EXPECTED[$tag]}" ]; then
        echo "  WARNING: $tag is now $got, expected ${EXPECTED[$tag]}." >&2
        echo "  The tag moved. The measurements in this repo's comments were" >&2
        echo "  taken against the expected commit; re-verify before trusting" >&2
        echo "  them against this content." >&2
    fi
    # The probes include the header under a VERSIONED name so one probe can be
    # built against several API versions in the same directory (probe.c does
    # exactly that via -DNVHDR).
    cp "$work/$tag/include/ffnvcodec/nvEncodeAPI.h" \
       "ffnvcodec/nvEncodeAPI_$tag.h"
done

# extbuf_probe.c needs the CUDA driver declarations; any of the three tags ships
# an identical copy, so take it from the one matching the driver.
cp "$work/n12.2.72.0/include/ffnvcodec/dynlink_cuda.h" ffnvcodec/dynlink_cuda.h

echo
echo "ffnvcodec/ ready:"
ls -1 ffnvcodec/
