# MinifyEag

Shrinks Eaglercraft client files (`.html` offline clients and `.epk` packs). Single Python file, with a window or a command line.

## What it does

1. **Audio**: every `.ogg` inside the EPK is re-encoded with Vorbis at the bitrate you choose (24 kbps recommended). Sounds already at or below the target are left alone. If a re-encode wouldn't come out smaller once compressed, the original audio is kept.
2. **Remove audio** (optional): deletes every `.ogg` from the client and fixes the EPK file count so the game still loads. The game runs silent.
3. **LZMA**: game JS, assets and lang packs are packed into LZMA streams grouped by kind. PNGs are stored as raw pixel rows (lossless) and rebuilt by the page. The page rebuilds an uncompressed EPK in memory.
4. **Text encoding**: the binary payload is written into the html as text.

| Encoding | Bits per char | Notes |
|----------|---------------|-------|
| `utf8`   | 7  | Works with any server or `fetch().text()` (+14%) |
| `utf16`  | 15 | File is saved as UTF-16 with a BOM (+7%) |
| `latin1` | 8  | Page is read as windows-1252 (+1.6%) |

`utf16` and `latin1` only work when the browser reads the file itself (opened from disk or served unchanged), not if a launcher re-reads it as UTF-8 text.

## Supported html layouts

- `data:application/octet-stream;base64,...` assets URI
- `var files = {...}` base64 table read by `decodeBase64ValueToU8()`
- `<script type="application/octet-stream" id="eag-inline-*">` base64 blocks (newer single-file clients). EPK blocks are re-encoded and written back in the same format; the brotli wasm blocks are left as they are.
- Large inline game scripts

## Requirements

- Python 3
- `numpy`
- `ffmpeg` with `libvorbis` (only needed for audio re-encoding)
- `tkinter` for the window (on Linux: `sudo apt install python3-tk`)

Put `ffmpeg` / `ffmpeg.exe` next to the script, or have it on your `PATH`.

## Usage

Window:

```
python EaglerFileMinifier.py
```

Command line:

```
python EaglerFileMinifier.py client.html
python EaglerFileMinifier.py a.html b.epk -k 24 -l 9 --max-rate 22050 --mono --encoding utf16
python EaglerFileMinifier.py client.html --strip-audio
```

Output is saved next to the original as `name.small.html`. Originals are never changed.

### Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file (single input only) |
| `-k, --kbps` | Audio bitrate (default 24) |
| `--max-rate` | Cap the audio sample rate, e.g. `22050` |
| `--mono` | Force mono audio |
| `--no-audio` | Skip audio re-encoding |
| `--strip-audio` | Remove all sounds (`.ogg`) from the client |
| `--no-deflate` | Store big assets uncompressed (no LZMA) |
| `--encoding` | `utf8` (default), `utf16` or `latin1` |
| `-l, --level` | Compression level 1-9 (default 9) |
| `-j, --jobs` | Worker threads |
| `--ffmpeg` | Path to ffmpeg |
| `--suffix` | Output name suffix (default `.small`) |
| `--gui` | Open the window |

## Notes

- Audio re-encoding is the slow part. A client with thousands of sounds can take several minutes on a single core.
- On clients that are already brotli-packed, the gains are limited: only the sounds can be shrunk.
- Always test the output in a browser. Removing audio and the `latin1`/`utf16` encodings change how the page loads.
