#!/usr/bin/env python3
"""Compare the probe's decoded NV12 output against the pattern it encoded.

A bitstream that NVENC returned NV_ENC_SUCCESS for proves only that the driver
accepted the call.  This checks whether it read the two planes where we put
them: bar-by-bar luma, and bar-by-bar interleaved chroma.
"""
import sys

W, H = 256, 128


def bt709_limited(r, g, b):
    kr, kb = 0.2126, 0.0722
    kg = 1.0 - kr - kb
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2.0 * (1.0 - kb))
    cr = (r - y) / (2.0 * (1.0 - kr))
    return (
        max(0, min(255, int(16.0 + 219.0 * y + 0.5))),
        max(0, min(255, int(128.0 + 224.0 * cb + 0.5))),
        max(0, min(255, int(128.0 + 224.0 * cr + 0.5))),
    )


BARS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
EXPECT = [bt709_limited(*c) for c in BARS]


def check(path):
    data = open(path, "rb").read()
    want = W * H * 3 // 2
    if len(data) != want:
        print(f"{path}: {len(data)} bytes, expected {want}")
        return False
    luma = data[: W * H]
    chroma = data[W * H:]

    ok = True
    print(f"\n{path}")
    print("  bar   expect Y/U/V    got Y/U/V (centre sample)   verdict")
    for i, (ey, eu, ev) in enumerate(EXPECT):
        # Centre of bar i, avoiding the edges where the encoder's own filtering
        # legitimately blends neighbouring bars.
        x = i * (W // 4) + (W // 8)
        y = H // 2
        gy = luma[y * W + x]
        cx, cy = x // 2, y // 2
        gu = chroma[cy * W + cx * 2]
        gv = chroma[cy * W + cx * 2 + 1]
        # H.264 at the driver's default QP is lossy; 6 codes is loose enough to
        # absorb that and far tighter than a wrong-plane or wrong-matrix error,
        # which move a channel by tens of codes.
        bad = abs(gy - ey) > 6 or abs(gu - eu) > 6 or abs(gv - ev) > 6
        ok &= not bad
        print(f"  {i}    {ey:3d}/{eu:3d}/{ev:3d}       {gy:3d}/{gu:3d}/{gv:3d}"
              f"                 {'MISMATCH' if bad else 'ok'}")
    print(f"  => {'PLANES READ CORRECTLY' if ok else 'WRONG PIXELS'}")
    return ok


if __name__ == "__main__":
    results = {p: check(p) for p in sys.argv[1:]}
    print("\n---- verdict ----")
    for p, r in results.items():
        print(f"{p}: {'ok' if r else 'FAILED'}")
