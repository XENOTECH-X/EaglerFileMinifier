# MinifyEag

Shrinks Eaglercraft client files. Works on offline `.html` clients and raw `.epk` asset packs.

## What it does

1. **Audio:** re-encodes every `.ogg` inside the EPK to 24 kbps Vorbis. Sample rate and channels are kept, and sounds already at or below 24 kbps are left alone.
2. **UTF-8 packing:** replaces the base64 assets in the HTML (about 33% bigger than the raw data) with a UTF-8 text encoding (about 14% bigger). The page decodes it at load time.

Typical result: **about 10-15% smaller**. Most of the saving comes from the UTF-8 packing; audio only helps if the client has lots of high-bitrate sounds.

## Requirements

| Needs | Install |
| --- | --- |
| Python 3 | [python.org](https://www.python.org/downloads/) |
| numpy | `pip install numpy` |
| ffmpeg with libvorbis | [ffmpeg.org](https://ffmpeg.org/download.html) or `winget install ffmpeg` |

Check ffmpeg is on your PATH: `ffmpeg -version`

## How to use

1. Put `MinifyEag.py` in the same folder as your `.html` or `.epk` file.
2. Open that folder in a terminal (cmd or PowerShell).
3. Run:

```
python MinifyEag.py client.html
```

or

```
python MinifyEag.py assets.epk
```

The result is saved next to the original as `client.small.html` or `assets.small.epk`. Your original file is never changed.

## Options

| Flag | Meaning | Default |
| --- | --- | --- |
| `-o`, `--output` | Output file name | `<name>.small.<ext>` |
| `-k`, `--kbps` | Audio bitrate | `24` |
| `-j`, `--jobs` | Parallel ffmpeg jobs | CPU count |

Example: `python MinifyEag.py client.html -k 32 -j 4`

## Supported formats

- HTML with a base64 `assetsURI` data URI
- HTML with a `var files = { ... }` chunk table (WASM-GC offline builds)
- Raw `.epk` files

**Note:** in WASM-GC `.epw` assets, audio is left untouched and only the UTF-8 packing is applied.

## Notes

- Keep `<meta charset="UTF-8">` near the top of the HTML, or the text blocks will not load. The tool warns if it is missing.
- Test the `.small` file in a browser before replacing your original.
