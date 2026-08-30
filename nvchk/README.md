# `nvchk/` — the standalone probes the FFI comments cite

Every "measured", "verified" or "probe showed" claim in `src/interop/` points at a
file in here. They are not part of the Cargo build and nothing in `cargo test`
runs them; they exist so those claims stay **falsifiable** rather than becoming
folklore the next session cites as established fact.

That is the whole rationale. A comment saying *"the driver ignores this field"*
with no reproducible probe behind it is indistinguishable from a guess after a
few months — and one such comment, generalised past the case it was measured on,
is what a `pitch = 0` NV12 registration turned into: a process kill with no error
return. See `nv12_pitch_probe.c`.

## Running them

```bash
cd nvchk
./fetch_headers.sh     # vendor headers, pinned by tag AND commit
./build_probes.sh      # plain gcc; no Cargo, no import stub, no delay-load
./probe_n12.2.72.0.exe # layout/enumerants — safe, touches no driver
./nv12_probe.exe               # encodes real frames; the last rung CAN kill the process
./nv12_probe.exe --codec hevc  # same, but an HEVC session + the HEVC GUID check
```

Requires MSYS2/MinGW `gcc` and an NVIDIA driver. `check_nv12.py` needs Python and
`ffmpeg` on PATH to decode a bitstream before checking it.

Only the `.c`, `.py` and `.sh` files are committed. `ffnvcodec/`, the `.exe`s and
all probe output are gitignored and regenerable — `ffnvcodec/` in particular is
~900 KB of third-party source that is not ours to redistribute, and pinning it by
tag *and* commit hash is a stronger provenance claim than a copy in our tree
(a copy can be edited; a tag mismatch is reported by `fetch_headers.sh`).

## What each probe establishes

| file | question | recorded in |
|---|---|---|
| `probe.c` | struct sizes, field offsets, enumerant values, `_VER` words, function-table slots — built once per API version (`n12.0.16.0`, `n12.2.72.0`, `n13.0.19.0`) so a constant is only called version-stable when all three agree | `src/interop/ffi/nvenc.rs` |
| `nv12_probe.c` | which NV12 allocation shape NVENC accepts: rung A linear buffer + `CUDADEVICEPTR`, rung B over-tall `CUarray`, rung B0 the same array at `pitch = 0`. Also, via `--codec h264\|hevc`, whether the CODEC GUID in `encode_interop.rs` is byte-for-byte the vendor header's — it initialises the session with the **Rust constant's bytes**, not the header's, so a typo there reproduces the real failure here | `src/interop/encode_interop.rs`, `src/interop/nv12_encode.rs` |
| `nv12_pitch_probe.c` | isolates the pitch question, one value per process so a crash cannot be blamed on a previous rung. Takes `abgr10` as a second arg to show the *old* single-plane result and the NV12 result side by side — the scope boundary the earlier comment crossed | `src/interop/encode_interop.rs` |
| `extbuf_probe.c` | `CUDA_EXTERNAL_MEMORY_*` layout; whether `D3D12_RESOURCE` is 4 or 5 and which the driver tolerates; that a mapped buffer frees with `cuMemFree`; logical-vs-padded import size | `src/interop/ffi/cuda_gl_vk_interop.rs`, `src/interop/external_buffer.rs` |
| `d3d12_buf_probe.c` | the shape that shipped: D3D12 committed BUFFER → NT handle → `cuImportExternalMemory` → `cuExternalMemoryGetMappedBuffer` → registered as `CUDADEVICEPTR`+NV12. Answers whether chroma is read at `pitch * height` or `width * height` by using `pitch != width` | `src/interop/ffi/nvenc.rs`, `src/interop/nv12_encode.rs` |
| `check_nv12.py` | decodes a probe's bitstream and compares each bar's centre sample against BT.709 limited-range codes computed independently in Python | all of the above |

`check_nv12.py` is the part that makes the rungs mean anything: **`NV_ENC_SUCCESS`
proves only that the driver accepted the call.** NVENC will happily encode
garbage, so a rung is "ok" here only when the *pixels* came back right.

## Re-verified 2026-08-30, after vendoring

Rebuilt from a fresh `fetch_headers.sh` (all three tags at their expected commits;
header content byte-identical to the originals modulo line endings) and re-ran:

- `probe_n12.2.72.0.exe` — API 12.2, `ABGR10 = 0x20000000`,
  `CUDAARRAY = 0x2`, `NV_ENC_REGISTER_RESOURCE` 1536 bytes,
  `NV_ENC_PIC_PARAMS` 3360, `..._VER = 0x7205000C` / `0xF207000C`. Matches the
  constants in `src/interop/ffi/nvenc.rs`.
- `extbuf_probe.exe layout` — `HANDLE_DESC` 104 (type 0, handle 8, size 24,
  flags 32), `BUFFER_DESC` 88, `MIPMAPPED_ARRAY_DESC` 120. Matches
  `cuda_gl_vk_interop.rs`.
- `nv12_probe.exe` — rung A `NV_ENC_SUCCESS`, 159-byte bitstream; decoded through
  `ffmpeg -pix_fmt nv12` and checked: all four bars exact
  (`63/102/240`, `173/42/26`, `32/240/118`, `235/128/128`), zero deviation.
  Rung B ok at 166 bytes. **Rung B0 (`pitch = 0`) still kills the process** —
  output stops after `nvEncMapInputResource` returns SUCCESS and no summary is
  printed. That is the documented behaviour, reproduced.

The pitch-0 kill is why `nv12_probe.exe` prints unbuffered and why the rungs are
ordered with it last.

## HEVC codec GUID — added 2026-08-30 (P1.2)

`nv12_probe.c --codec hevc` was added after zero-copy NVENC turned out to be
**unreachable for every H.265 export**. `NV_ENC_CODEC_HEVC_GUID` in
`src/interop/encode_interop.rs` read `88 CD 0C 79 …` where `{790CDC88-…}`
little-endian is `88 DC 0C 79 …` — one transposed nibble. The whole symptom was
`nvEncInitializeEncoder` returning `NV_ENC_ERR_UNSUPPORTED_PARAM` (12), which
names no field and is indistinguishable from "this GPU has no HEVC encoder", so
`VideoEncoderBackend::select` logged a warning and silently used the FFmpeg
encoder instead.

Two design points make the probe mean something:

- The comparison is on **raw memory bytes**, not the `{Data1-Data2-…}` text form.
  `Data1` is a `DWORD`, so a byte array that "looks like" the printed GUID is
  wrong — which is exactly how the typo survived review.
- The session is initialised with the **Rust constant's bytes**, copied in with
  `memcpy`, not with the header's `NV_ENC_CODEC_HEVC_GUID`. Using the header
  would let the driver call succeed while the shipping constant was still wrong,
  and the probe would have reported a mismatch alongside a green driver result.

Measured on RTX 3050, driver API 12.2:

- `--codec hevc`: byte diff **MATCH**, `nvEncInitializeEncoder` →
  `NV_ENC_SUCCESS`, rungs A and B write 179-byte bitstreams.
  `ffmpeg -f hevc -i rung_a.hevc -pix_fmt nv12` decoded and checked with
  `check_nv12.py`: all four bars exact (`63/102/240`, `173/42/26`, `32/240/118`,
  `235/128/128`), zero deviation — so HEVC reads the two planes the same way
  H.264 does.
- **Negative control**, one nibble flipped back to `0xCD` in a scratch copy: byte
  diff reports `*** MISMATCH ***` at index 1 (`header 0xdc vs Rust 0xcd`) and
  `nvEncInitializeEncoder` → `NV_ENC_ERR_UNSUPPORTED_PARAM (12)`, exit 1. The
  original failure, reproduced on demand.
- `--codec h264`: unchanged from the 2026-08-30 run above.
