#!/usr/bin/env python3
"""
MinifyEag - shrink Eaglercraft client files (.html offline clients and .epk packs).

  python MinifyEag.py                      open the window (select files, tweak settings)
  python MinifyEag.py client.html          command line, recommended settings
  python MinifyEag.py a.html b.epk -k 24 -l 9 --max-rate 22050 --mono --encoding utf16

What it does
  1. audio    every .ogg inside the EPK is re-encoded with Vorbis at the chosen bitrate
              (24 kbps recommended). Sounds already small enough are left alone, and the
              original audio is kept if the re-encode wouldn't be smaller once compressed.
  2. lzma     game JS, assets and lang packs are packed into LZMA streams grouped by kind
              (pixels / audio / text / binary). PNGs are stored as raw pixel rows (lossless,
              pixel-exact) and rebuilt by the page, which LZMA squeezes ~25% harder than deflate.
              The page rebuilds an uncompressed EPK in memory (CRCs recomputed).
  3. text     the binary payload is written into the html as text:
                utf8   7 bits/char, works with any server or fetch().text()  (+14%)
                utf16  15 bits/char, file saved as UTF-16 with a BOM          (+7%)
                latin1 8 bits/char, page read as windows-1252                 (+1.6%)
              utf16/latin1 only work when the browser reads the file itself (opened from
              disk or served unchanged), not if something re-reads it as UTF-8 text.

needs: python 3, numpy, ffmpeg with libvorbis (only for the audio step)
"""
import argparse, base64, fnmatch, lzma, os, queue, re, shutil, struct, subprocess, sys, threading, zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import numpy as np

EPK_MAGIC = b"EAGPKG$$"
ILLEGAL = (0, 10, 13, 60)  # NUL, LF, CR, '<' -> not safe raw inside a <script> block
RECOMMENDED_KBPS = 24
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Cancelled(Exception):
    pass


@dataclass
class Opts:
    audio: bool = True
    bundle: bool = False       # gzip the whole finished page and store it as latin1 text behind a tiny loader
    strip_audio: bool = False  # drop every .ogg from the EPKs (no sound at all)
    drop: list = field(default_factory=list)  # glob patterns of EPK entries to remove
    langs: str = ""            # keep only these language files (comma separated, e.g. en_US); empty = keep all
    kbps: int = RECOMMENDED_KBPS
    max_rate: int = 0          # 0 = keep original sample rate
    mono: bool = False
    deflate: bool = True
    level: int = 9             # 1 (fast) .. 9 (smallest)
    encoding: str = "utf8"     # how the binary payload is written into the html: utf8 | utf16 | latin1
    jobs: int = os.cpu_count() or 4
    blobs: list = field(default_factory=list)
    plain: object = None
    ffmpeg: str = "ffmpeg"
    cancel: threading.Event = field(default_factory=threading.Event)


# ---------------------------------------------------------------- ffmpeg ----
def find_ffmpeg():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "ffmpeg.exe"), os.path.join(here, "ffmpeg")):
        if os.path.isfile(c):
            return c
    return shutil.which("ffmpeg") or "ffmpeg"


def ffmpeg_status(path):
    """(ok, message)"""
    try:
        r = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True, creationflags=NOWIN, timeout=20)
    except Exception:
        return False, "ffmpeg not found"
    if b"libvorbis" not in r.stdout:
        return False, "ffmpeg found but has no libvorbis"
    return True, "ffmpeg ready"


# ---------------------------------------------------------------- EPK v2 ----
def epk_parse(raw):
    assert raw[:8] == EPK_MAGIC, "not an EPK"
    p = 8
    for _ in range(2):  # version string, filename
        p += 1 + raw[p]
    p += 2 + struct.unpack(">H", raw[p:p + 2])[0]  # comment
    p += 8 + 4  # timestamp, file count
    head, comp = raw[:p], raw[p:p + 1]
    body = raw[p + 1:]
    if comp == b"G":
        d = zlib.decompressobj(31); inner = d.decompress(body); footer = d.unused_data
    elif comp == b"Z":
        d = zlib.decompressobj(15); inner = d.decompress(body); footer = d.unused_data
    elif comp == b"0":
        inner, footer = body, b""
    else:
        raise ValueError("unknown EPK compression %r" % comp)
    ents, p = [], 0
    while True:
        t = inner[p:p + 4]; p += 4
        if t == b"END$":
            break
        nl = inner[p]; p += 1
        name = inner[p:p + nl].decode(); p += nl
        ln = struct.unpack(">I", inner[p:p + 4])[0]; p += 4
        if t == b"FILE":
            data = inner[p + 4:p + ln - 1]; p += ln
        else:
            data = inner[p:p + ln]; p += ln
        assert inner[p:p + 1] == b">", "bad EPK entry " + name
        p += 1
        ents.append([t, name, data])
    return head, comp, ents, inner[p:], footer


def epk_build(head, comp, ents, tail, footer, o):
    out = bytearray()
    for t, n, x in ents:
        nb = n.encode()
        out += t + bytes([len(nb)]) + nb
        if t == b"FILE":
            out += struct.pack(">II", len(x) + 5, zlib.crc32(x)) + x + b":"
        else:
            out += struct.pack(">I", len(x)) + x
        out += b">"
    out += b"END$"
    if comp == b"0" and o.deflate:  # uncompressed -> gzip (client reads it natively)
        comp, footer, tail = b"G", tail + footer, b""
    out += tail
    if comp in (b"G", b"Z"):
        c = zlib.compressobj(o.level, zlib.DEFLATED, 31 if comp == b"G" else 15)
        body = c.compress(bytes(out)) + c.flush()
    else:
        body = bytes(out)
    return head + comp + body + footer


def epk_filter(head, ents, o):
    """remove the entries the options ask to drop (all .ogg, glob patterns, unwanted .lang files) and
    fix the file count stored at the end of the EPK header. returns (head, entries, removed)"""
    pats = list(o.drop)
    langs = {x.strip().lower() for x in o.langs.replace(";", ",").split(",") if x.strip()}
    keep = []
    for e in ents:
        n = e[1]; low = n.lower()
        gone = o.strip_audio and low.endswith(".ogg")
        gone = gone or any(fnmatch.fnmatchcase(n, p) for p in pats)
        if not gone and langs and e[0] == b"FILE" and low.endswith(".lang"):
            gone = low.rsplit("/", 1)[-1][:-5] not in langs
        if not gone:
            keep.append(e)
    removed = len(ents) - len(keep)
    if removed:
        cnt = struct.unpack(">I", head[-4:])[0]
        head = head[:-4] + struct.pack(">I", cnt - removed)
    return head, keep, removed


# ------------------------------------------------------------------ audio ---
def ogg_info(data):
    """(sample_rate, channels, kbps) read straight from the ogg/vorbis headers, or None"""
    i = data.find(b"\x01vorbis")
    j = data.rfind(b"OggS")
    if i < 0 or j < 0:
        return None
    ch = data[i + 11]; sr = struct.unpack("<I", data[i + 12:i + 16])[0]
    gp = struct.unpack("<q", data[j + 6:j + 14])[0]
    if sr <= 0 or gp <= 0:
        return None
    return sr, ch, len(data) * 8 / (gp / sr) / 1000


def to_vorbis(data, o):
    """returns (status, result): ("skip"|"fail"|"ok", data / error text)"""
    if o.cancel.is_set():
        return "skip", None
    info = ogg_info(data)
    if info is None:
        return "skip", None
    sr, ch, kbps = info
    sr2 = min(sr, o.max_rate) if o.max_rate else sr
    ch2 = 1 if o.mono else ch
    if kbps <= o.kbps and sr2 == sr and ch2 == ch:  # already at/below target: leave alone
        return "skip", None
    r = subprocess.run(
        [o.ffmpeg, "-v", "error", "-i", "pipe:0", "-vn", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
         "-c:a", "libvorbis", "-b:a", "%dk" % o.kbps, "-ar", str(sr2), "-ac", str(ch2),
         "-f", "ogg", "pipe:1"],
        input=data, capture_output=True, creationflags=NOWIN)
    if r.returncode == 0 and r.stdout:
        return "ok", r.stdout
    return "fail", r.stderr.decode("utf8", "replace").strip()[:200]


def shrink_audio(ents, o, log, sub):
    idx = [i for i, e in enumerate(ents) if e[0] == b"FILE" and e[1].lower().endswith(".ogg")]
    if not idx:
        return 0
    ok, msg = ffmpeg_status(o.ffmpeg)
    if not ok:
        log("  audio skipped: " + msg)
        return 0
    done = [0]
    def work(i):
        r = to_vorbis(ents[i][2], o)
        done[0] += 1
        sub(0.8 * done[0] / len(idx))
        return r
    with ThreadPoolExecutor(o.jobs) as ex:
        outs = list(ex.map(work, idx))
    if o.cancel.is_set():
        raise Cancelled()
    before = after = changed = skipped = failed = bigger = 0
    err = ""
    for i, (st, new) in zip(idx, outs):
        old = ents[i][2]; before += len(old)
        if st == "ok" and len(new) < len(old):
            ents[i][2] = new; after += len(new); changed += 1
        else:
            after += len(old)
            if st == "skip": skipped += 1
            elif st == "fail": failed += 1; err = err or new
            else: bigger += 1
    log("  audio: %d ogg, %.2f MB -> %.2f MB  (%d re-encoded, %d already at/below target, %d not smaller, %d failed)"
        % (len(idx), before / 1e6, after / 1e6, changed, skipped, bigger, failed))
    if failed:
        log("  ffmpeg error: " + err)
    return changed


def rebuild_epk(raw, o, log, sub):
    """audio shrink + (optional) gzip. Never returns something bigger than the original EPK:
    gzip packs the original ogg files tightly (they share identical headers), so a smaller
    re-encode can still end up bigger once compressed - in that case the audio is left alone."""
    head, comp, ents, tail, footer = epk_parse(raw)
    head, ents, removed = epk_filter(head, ents, o)
    if removed:
        log("  removed %d files" % removed)
    if o.strip_audio:
        return epk_build(head, comp, ents, tail, footer, o) if removed else raw
    originals = [e[2] for e in ents]
    changed = shrink_audio(ents, o, log, sub) if o.audio else 0
    if not changed:
        return epk_build(head, comp, ents, tail, footer, o) if ((comp == b"0" and o.deflate) or removed) else raw
    new = epk_build(head, comp, ents, tail, footer, o)
    if comp in (b"G", b"Z") and not removed:
        base = raw
    else:
        for e, d in zip(ents, originals):
            e[2] = d
        base = epk_build(head, comp, ents, tail, footer, o)
    if len(new) < len(base):
        log("  audio re-encode saves %.2f MB after compression" % ((len(base) - len(new)) / 1e6))
        return new
    log("  audio re-encode would not shrink this EPK after compression - kept original audio")
    return base


# ------------------------------------------------------------------ LZMA ----
STORED = 255  # props byte meaning "not compressed"
# (lc, lp, pb) tried per stream class (the smallest result wins at level 9, the first one otherwise)
CANDS = {
    "meta": [(3, 0, 0)],
    "js":   [(4, 0, 0), (3, 0, 0)],
    "px4":  [(0, 2, 2), (0, 0, 0)],
    "pxo":  [(3, 0, 0), (0, 0, 0)],
    "ogg":  [(3, 0, 0), (0, 0, 0)],
    "txt":  [(3, 0, 0), (4, 0, 0)],
    "bin":  [(3, 0, 0), (0, 0, 0)],
}
TEXT_EXT = {"json", "lang", "fsh", "vsh", "glsl", "html", "htm", "txt", "css", "csv", "properties", "mcmeta",
            "wlist", "js", "xml", "cfg", "obj", "mtl", "svg", "md", "ini", "toml", "yml", "yaml", "frag", "vert", "shader"}


def varint(n):
    out = bytearray()
    while True:
        b = n & 127; n >>= 7
        out.append(b | 128 if n else b)
        if not n:
            return bytes(out)


def lz_encode(data, lc, lp, pb, level):
    n = max(len(data), 1)
    flt = dict(id=lzma.FILTER_LZMA1, dict_size=min(1 << 25, max(1 << 16, 1 << (n - 1).bit_length())), lc=lc, lp=lp, pb=pb)
    if level >= 9:
        flt.update(preset=9 | lzma.PRESET_EXTREME, nice_len=273, mf=lzma.MF_BT4, depth=1000)
    elif level >= 7:
        flt.update(preset=9, nice_len=273, mf=lzma.MF_BT4, depth=0)
    else:
        flt.update(preset=max(level, 1))
    return lzma.compress(data, format=lzma.FORMAT_RAW, filters=[flt])


def make_stream(data, cls, o):
    """-> (props, outlen, compressed bytes) for one stream of the container"""
    data = bytes(data)
    if not o.deflate or not data:
        return STORED, len(data), data
    cands = CANDS[cls] if o.level >= 9 else CANDS[cls][:1]
    best = None
    for lc, lp, pb in cands:
        c = lz_encode(data, lc, lp, pb, o.level)
        if best is None or len(c) < len(best[1]):
            best = ((pb * 5 + lp) * 9 + lc, c)
    if len(best[1]) >= len(data):
        return STORED, len(data), data
    return best[0], len(data), best[1]


def container(streams):
    out = bytearray()
    for props, n, c in streams:
        out += bytes([props]) + varint(n) + varint(len(c)) + c
    return bytes(out)


# ------------------------------------------------------------------ PNG -----
PNG_SIG = b"\x89PNG\r\n\x1a\n"
PNG_CH = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def png_unfilter(raw, h, stride, bpp):
    """raw = h rows of (filter byte + stride bytes) -> (h, stride) uint8 array of unfiltered rows, or None"""
    img = np.zeros((h, stride), dtype=np.uint8)
    ft = []
    prev = np.zeros(stride, dtype=np.uint8)
    p = 0
    for y in range(h):
        f = raw[p]
        if f > 4:
            return None, None
        row = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=p + 1).copy()
        p += 1 + stride
        if f == 1:
            for c in range(bpp):
                row[c::bpp] = np.cumsum(row[c::bpp], dtype=np.uint64).astype(np.uint8)
        elif f == 2:
            row = (row + prev).astype(np.uint8)
        elif f == 3:
            r = row.tolist(); pv = prev.tolist(); out = [0] * stride
            for i in range(stride):
                a = out[i - bpp] if i >= bpp else 0
                out[i] = (r[i] + ((a + pv[i]) >> 1)) & 255
            row = np.array(out, dtype=np.uint8)
        elif f == 4:
            r = row.tolist(); pv = prev.tolist(); out = [0] * stride
            for i in range(stride):
                a = out[i - bpp] if i >= bpp else 0
                b = pv[i]; c = pv[i - bpp] if i >= bpp else 0
                pa = abs(b - c); pb = abs(a - c); pc = abs(a + b - 2 * c)
                out[i] = (r[i] + (a if (pa <= pb and pa <= pc) else (b if pb <= pc else c))) & 255
            row = np.array(out, dtype=np.uint8)
        img[y] = row; ft.append(f); prev = row
    return img, ft


def png_refilter_ok(img, ft, raw, h, stride, bpp):
    """re-apply every row's filter to the unfiltered image and compare with the original bytes"""
    zero = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        cur = img[y].astype(np.int32)
        up = img[y - 1].astype(np.int32) if y else zero
        a = np.concatenate([np.zeros(bpp, dtype=np.int32), cur[:-bpp]]) if stride > bpp else np.zeros(stride, dtype=np.int32)
        c = np.concatenate([np.zeros(bpp, dtype=np.int32), up[:-bpp]]) if stride > bpp else np.zeros(stride, dtype=np.int32)
        f = ft[y]
        if f == 0: pred = 0
        elif f == 1: pred = a
        elif f == 2: pred = up
        elif f == 3: pred = (a + up) >> 1
        else:
            pa = np.abs(up - c); pb = np.abs(a - c); pc = np.abs(a + up - 2 * c)
            pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, up, c))
        want = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=y * (stride + 1) + 1)
        if not np.array_equal(((cur - pred) & 255).astype(np.uint8), want):
            return False
    return True


def png_split(d):
    """-> (cls, pre, rows_bytes, suf) for a plain non-interlaced PNG, else None.
    pre/suf are every byte before/after the IDAT chunks (kept verbatim); the pixel rows are stored unfiltered."""
    try:
        if d[:8] != PNG_SIG:
            return None
        p, chunks = 8, []
        while p + 12 <= len(d):
            ln = struct.unpack(">I", d[p:p + 4])[0]
            if p + 12 + ln > len(d):
                return None
            chunks.append((p, d[p + 4:p + 8], ln)); p += 12 + ln
        if p != len(d) or len(chunks) < 3 or chunks[0][1:] != (b"IHDR", 13) or chunks[-1][1] != b"IEND":
            return None
        idx = [i for i, c in enumerate(chunks) if c[1] == b"IDAT"]
        if not idx or idx[-1] - idx[0] != len(idx) - 1:
            return None
        w, h, depth, ct, cm, fm, il = struct.unpack(">IIBBBBB", d[16:29])
        if il or cm or fm or ct not in PNG_CH or depth not in (1, 2, 4, 8, 16) or not w or not h:
            return None
        bits = depth * PNG_CH[ct]; stride = (w * bits + 7) // 8; bpp = max(1, bits // 8)
        z = b"".join(d[c[0] + 8:c[0] + 8 + c[2]] for c in chunks[idx[0]:idx[-1] + 1])
        dco = zlib.decompressobj()
        raw = dco.decompress(z)
        if not dco.eof or len(raw) != h * (stride + 1):
            return None
        img, ft = png_unfilter(raw, h, stride, bpp)
        if img is None or not png_refilter_ok(img, ft, raw, h, stride, bpp):
            return None
        pre = d[:chunks[idx[0]][0]]
        suf = d[chunks[idx[-1]][0] + 12 + chunks[idx[-1]][2]:]
        return ("px4" if (depth == 8 and ct == 6) else "pxo"), pre, img.tobytes(), suf, w, h, stride
    except Exception:
        return None


def png_size(pre, suf, h, stride):
    f = h * (stride + 1)
    return len(pre) + 12 + 2 + f + 5 * (-(-f // 65535)) + 4 + len(suf)


# ------------------------------------------------------------ EPK -> streams -
def classify(name):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return "ogg" if ext == "ogg" else ("txt" if ext in TEXT_EXT else "bin")


def pack_epk(raw, o, log, sub):
    """EPK -> container: PNGs are stored as unfiltered pixel rows, everything grouped by kind into LZMA streams.
    The page rebuilds an (uncompressed) EPK in memory."""
    head, comp, ents, tail, footer = epk_parse(raw)
    tail0 = tail + footer
    head, ents, removed = epk_filter(head, ents, o)
    if removed:
        log("  removed %d files" % removed)
    originals = [e[2] for e in ents]
    changed = shrink_audio(ents, o, log, sub) if o.audio else 0
    order = ["px4", "pxo", "ogg", "txt", "bin"]

    def layout(ents):
        parts, bufs = [], {c: bytearray() for c in order}
        npng = 0
        for t, n, d in ents:
            if t == b"FILE":
                sp = png_split(d) if n.lower().endswith(".png") else None
                if sp:
                    cls, pre, rows, suf, w, h, stride = sp
                    bufs[cls] += rows
                    parts.append((t, n, 3, cls, (pre, suf, len(rows)), png_size(pre, suf, h, stride) + 9))
                    npng += 1
                else:
                    cls = classify(n)
                    bufs[cls] += d
                    parts.append((t, n, 2, cls, (len(d),), 4 + 4 + len(d) + 1))
            else:
                bufs["bin"] += d
                parts.append((t, n, 0, "bin", (len(d),), 4 + len(d)))
        return parts, bufs, npng

    parts, bufs, npng = layout(ents)
    streams = {c: make_stream(bufs[c], c, o) for c in order if bufs[c]}
    if changed:  # keep the original audio if re-encoding doesn't pay off once compressed
        parts2, bufs2, _ = layout([[t, n, d] for (t, n, _), d in zip(ents, originals)])
        orig = make_stream(bufs2["ogg"], "ogg", o)
        if len(orig[2]) <= len(streams["ogg"][2]):
            log("  audio re-encode would not shrink this EPK after compression - kept original audio")
            parts, bufs, streams["ogg"] = parts2, bufs2, orig
        else:
            log("  audio re-encode saves %.2f MB after compression" % ((len(orig[2]) - len(streams["ogg"][2])) / 1e6))
    idx = {c: i + 1 for i, c in enumerate(c for c in order if c in streams)}
    total = len(head) + 1 + sum(4 + 1 + len(n.encode()) + 1 + p[5] for (t, n, *_), p in zip(ents, parts)) + 4 + len(tail0)
    meta = bytearray(b"\x01") + varint(total) + varint(len(head)) + head + varint(len(tail0)) + tail0 + varint(len(parts))
    for (t, n, k, cls, extra, _) in parts:
        nb = n.encode()
        meta += t + bytes([len(nb)]) + nb + bytes([k]) + varint(idx[cls])
        if k == 3:
            pre, suf, nrows = extra
            meta += varint(len(pre)) + pre + varint(len(suf)) + suf + varint(nrows)
        else:
            meta += varint(extra[0])
    sm = make_stream(meta, "meta", o)
    out = container([sm] + [streams[c] for c in order if c in streams])
    log("  epk: %d entries (%d png), %s" % (len(parts), npng, ", ".join("%s %.2f->%.2f MB" % (c, len(bufs[c]) / 1e6, len(streams[c][2]) / 1e6) for c in order if c in streams)))
    return out


def pack_raw(data, cls, o):
    return container([(STORED, 1, b"\x00"), make_stream(data, cls, o)])


# ------------------------------------------------------- text encodings -----
def _emit(code):
    ln = np.where(code < 128, 1, 2).astype(np.int64)
    end = np.cumsum(ln); start = end - ln
    out = np.empty(int(end[-1]), dtype=np.uint8)
    one = ln == 1
    out[start[one]] = code[one]
    two = ~one
    out[start[two]] = 0xC0 | (code[two] >> 6)
    out[start[two] + 1] = 0x80 | (code[two] & 63)
    return out.tobytes()


def utf8_pack(data):
    """7 bits per byte. Values in ILLEGAL are emitted as one 2-byte UTF-8 char
    (carrying the illegal value's index + the next 7 bits)."""
    n = len(data)
    buf = np.frombuffer(data + b"\0" * (-n % 7) + b"\xff" * 7, dtype=np.uint8).reshape(-1, 7)
    b0, b1, b2, b3, b4, b5, b6 = (buf[:, i] for i in range(7))
    v = np.empty((len(buf), 8), dtype=np.uint8)  # eight 7-bit groups per 7 bytes
    v[:, 0] = b0 >> 1
    v[:, 1] = ((b0 & 1) << 6) | (b1 >> 2)
    v[:, 2] = ((b1 & 3) << 5) | (b2 >> 3)
    v[:, 3] = ((b2 & 7) << 4) | (b3 >> 4)
    v[:, 4] = ((b3 & 15) << 3) | (b4 >> 5)
    v[:, 5] = ((b4 & 31) << 2) | (b5 >> 6)
    v[:, 6] = ((b5 & 63) << 1) | (b6 >> 7)
    v[:, 7] = b6 & 127
    v = v.ravel()
    del buf, b0, b1, b2, b3, b4, b5, b6
    code = v.astype(np.uint16)
    drop = np.zeros(len(v), dtype=bool)
    lut = {x: k for k, x in enumerate(ILLEGAL)}
    last = -2
    for i in np.flatnonzero(np.isin(v, ILLEGAL)).tolist():
        if i == last + 1:  # was consumed as the partner of the previous lead
            continue
        code[i] = 128 + (lut[int(v[i])] << 7 | int(v[i + 1]))
        drop[i + 1] = True
        last = i
    code = code[~drop]
    step = 1 << 22  # emit in slices to keep memory low
    return b"".join(_emit(code[j:j + step]) for j in range(0, len(code), step))


LATIN_ESC = (0, 13, 60, 1)  # NUL, CR, '<' and the escape byte itself (written as 0x01, 0x30+index)


def latin1_pack(data):
    """8 bits per byte (page must be read as windows-1252 / latin-1). 4 byte values get a 2-byte escape."""
    a = np.frombuffer(data, dtype=np.uint8)
    esc = np.isin(a, LATIN_ESC)
    ln = np.where(esc, 2, 1).astype(np.int64)
    end = np.cumsum(ln); start = end - ln
    out = np.empty(int(end[-1]) if len(a) else 0, dtype=np.uint8)
    out[start[~esc]] = a[~esc]
    e = np.flatnonzero(esc)
    out[start[e]] = 1
    lut = np.zeros(256, dtype=np.uint8)
    for k, v in enumerate(LATIN_ESC):
        lut[v] = 48 + k
    out[start[e] + 1] = lut[a[e]]
    return out.tobytes()


def utf16_pack(data):
    """15 bits per UTF-16 code unit (page is saved as UTF-16LE with a BOM). Returns the raw little-endian bytes."""
    w = (1 << np.arange(14, -1, -1)).astype(np.uint32)
    step = 15 << 18  # bytes per slice (multiple of 15) to keep memory low
    out = []
    for j in range(0, len(data), step):
        part = data[j:j + step]
        part += b"\0" * (-len(part) % 15)
        bits = np.unpackbits(np.frombuffer(part, dtype=np.uint8)).reshape(-1, 15)
        out.append((bits.astype(np.uint32) @ w + 256).astype("<u2").tobytes())
    return b"".join(out)


# --------------------------------------------------------------- decoder ----
DEC_LZMA = r"""function eaglerLzma(S,q,n,pr){
var lc=pr%9,lp=(pr=(pr/9)|0)%5,pb=(pr/5)|0,O=new Uint8Array(n),o=0,R=4294967295,C=0,i,
M=function(k){return new Uint16Array(k).fill(1024)},
IM=M(192),IR=M(12),G0=M(12),G1=M(12),G2=M(12),I0=M(192),PS=M(256),PD=M(128),AL=M(16),
LC=M(2),LL=M(128),LM=M(128),LH=M(256),RC=M(2),RL=M(128),RM=M(128),RH=M(256),LT=M(768<<(lc+lp));
for(q++,i=0;i<4;i++)C=((C<<8)|S[q++])>>>0;
function N(){if(R<16777216){R=(R<<8)>>>0;C=((C<<8)|S[q++])>>>0}}
function B(A,k){var v=A[k],b=(R>>>11)*v,r;
if(C<b){R=b;A[k]=v+((2048-v)>>>5);r=0}else{R-=b;C-=b;A[k]=v-(v>>>5);r=1}
N();return r}
function T(A,k,c){var m=1;while(c--)m=(m<<1)|B(A,k+m);return m}
function V(A,k,c){var m=1,s=0,j;for(j=0;j<c;j++){var b=B(A,k+m);m=(m<<1)|b;s|=b<<j}return s}
function D(c){var r=0;while(c--){R>>>=1;var b=0;if(C>=R){C-=R;b=1}r=(r<<1)|b;N()}return r>>>0}
var st=0,a=0,b1=0,c1=0,d1=0,l,e,ps,y,z,pv,mb,mt,base,sy,sl,nb;
while(o<n){
ps=o&((1<<pb)-1);
if(!B(IM,(st<<4)+ps)){
pv=o?O[o-1]:0;base=768*(((o&((1<<lp)-1))<<lc)+(pv>>>(8-lc)));sy=1;
if(st>=7){mb=O[o-a-1];do{mt=(mb>>>7)&1;mb<<=1;y=B(LT,base+((1+mt)<<8)+sy);sy=(sy<<1)|y;if(mt!=y)break}while(sy<256)}
while(sy<256)sy=(sy<<1)|B(LT,base+sy);
O[o++]=sy;st=st<4?0:st<10?st-3:st-6;continue}
if(B(IR,st)){
if(!B(G0,st)){if(!B(I0,(st<<4)+ps)){st=st<7?9:11;O[o]=O[o-a-1];o++;continue}}
else{if(!B(G1,st))z=b1;else{if(!B(G2,st))z=c1;else{z=d1;d1=c1}c1=b1}b1=a;a=z}
st=st<7?8:11;
l=B(RC,0)?(B(RC,1)?16+T(RH,0,8)-256:8+T(RM,ps<<3,3)-8):T(RL,ps<<3,3)-8;
}else{
d1=c1;c1=b1;b1=a;st=st<7?7:10;
l=B(LC,0)?(B(LC,1)?16+T(LH,0,8)-256:8+T(LM,ps<<3,3)-8):T(LL,ps<<3,3)-8;
sl=T(PS,(l<4?l:3)<<6,6)-64;
if(sl<4)a=sl;else{nb=(sl>>1)-1;a=(2|(sl&1))<<nb;
if(sl<14)a+=V(PD,a-sl,nb);else{a+=D(nb-4)<<4;a+=V(AL,0,4)}}
a>>>=0;if(a==4294967295)break}
for(l+=2;l--&&o<n;o++)O[o]=O[o-a-1]}
return O}
"""

DEC_CORE = r"""var eaglerT;
function eaglerCrc(b,s,e){var T=eaglerT,c=-1,n,k;
if(!T){T=eaglerT=new Int32Array(256);for(n=0;n<256;n++){c=n;for(k=0;k<8;k++)c=c&1?-306674912^(c>>>1):c>>>1;T[n]=c}c=-1}
for(;s<e;s++)c=T[(c^b[s])&255]^(c>>>8);return ~c>>>0}
function eaglerW(O,o,v){O[o]=v>>>24;O[o+1]=v>>>16;O[o+2]=v>>>8;O[o+3]=v}
function eaglerV(A){var r=0,m=1,c;do{c=A[A.p++];r+=(c&127)*m;m*=128}while(c&128);return r}
function eaglerPng(O,o,pre,raw,suf,h,rb){
var F=h*(rb+1),nb=Math.ceil(F/65535),z=F+2+5*nb+4,T=new Uint8Array(F),i,j,l,a=1,b=0,e,c0;
for(i=0;i<h;i++)T.set(raw.subarray(i*rb,i*rb+rb),i*(rb+1)+1);
O.set(pre,o);o+=pre.length;eaglerW(O,o,z);o+=4;c0=o;
O[o++]=73;O[o++]=68;O[o++]=65;O[o++]=84;O[o++]=120;O[o++]=1;
for(j=0;j<F;j+=65535){l=Math.min(65535,F-j);O[o++]=j+l>=F?1:0;O[o++]=l&255;O[o++]=l>>8;O[o++]=~l&255;O[o++]=(~l>>8)&255;O.set(T.subarray(j,j+l),o);o+=l}
for(i=0;i<F;){for(e=Math.min(i+5552,F);i<e;i++){a+=T[i];b+=a}a%=65521;b%=65521}
eaglerW(O,o,((b<<16)|a)>>>0);o+=4;eaglerW(O,o,eaglerCrc(O,c0,o));o+=4;
O.set(suf,o);return o+suf.length}
function eaglerDecode(id,len){
var d=eaglerRead(id,len),S=[],Q=[],pr,n,m,i,M,O,o,ne,k,s,l,nl,h,pl,sl,pre,suf,c0;
for(d.p=0;d.p<len;){pr=d[d.p++];n=eaglerV(d);m=eaglerV(d);S.push(pr==255?d.subarray(d.p,d.p+n):eaglerLzma(d,d.p,n,pr));Q.push(0);d.p+=m}
M=S[0];M.p=0;if(!M[M.p++])return S[1];
O=new Uint8Array(eaglerV(M));l=eaglerV(M);O.set(M.subarray(M.p,M.p+l));o=l;M.p+=l;O[o++]=48;
l=eaglerV(M);var tl=M.subarray(M.p,M.p+l);M.p+=l;ne=eaglerV(M);
for(i=0;i<ne;i++){
O.set(M.subarray(M.p,M.p+4),o);o+=4;nl=M[M.p+4];O[o++]=nl;O.set(M.subarray(M.p+5,M.p+5+nl),o);o+=nl;M.p+=5+nl;
k=M[M.p++];s=eaglerV(M);
if(k==3){
pl=eaglerV(M);pre=M.subarray(M.p,M.p+pl);M.p+=pl;sl=eaglerV(M);suf=M.subarray(M.p,M.p+sl);M.p+=sl;l=eaglerV(M);
h=((pre[20]<<24)|(pre[21]<<16)|(pre[22]<<8)|pre[23])>>>0;
var F=h*(l/h+1),tot=pl+12+2+F+5*Math.ceil(F/65535)+4+sl;
eaglerW(O,o,tot+5);o+=8;c0=o;o=eaglerPng(O,o,pre,S[s].subarray(Q[s],Q[s]+l),suf,h,l/h);Q[s]+=l;
eaglerW(O,c0-4,eaglerCrc(O,c0,o));O[o++]=58}
else{l=eaglerV(M);
if(k==2){eaglerW(O,o,l+5);o+=8;O.set(S[s].subarray(Q[s],Q[s]+l),o);eaglerW(O,o-4,eaglerCrc(O,o,o+l));o+=l;O[o++]=58}
else{eaglerW(O,o,l);o+=4;O.set(S[s].subarray(Q[s],Q[s]+l),o);o+=l}
Q[s]+=l}
O[o++]=62}
O.set([69,78,68,36],o);o+=4;O.set(tl,o);return O}
function eaglerRunJS(id,len){var s=document.createElement("script");s.text=new TextDecoder("utf-8",{ignoreBOM:true}).decode(eaglerDecode(id,len));(document.head||document.documentElement).appendChild(s);s.remove()}
function eaglerEPK(id,len){return URL.createObjectURL(new Blob([eaglerDecode(id,len)],{type:"application/octet-stream"}))}
"""

DEC_READ = {
    # 7 bits per char; chars >= 128 stand for one of the four illegal values + the next 7 bits
    "utf8": r"""function eaglerRead(id,len){
var e=document.getElementById(id),s=e.textContent,o=new Uint8Array(len),j=0,a=0,n=0,I=[0,10,13,60],i,l,c;
function put(v){a=((a<<7)|v)&16383;n+=7;if(n>=8){n-=8;if(j<len)o[j++]=(a>>n)&255}}
for(i=0,l=s.length;i<l;i++){c=s.charCodeAt(i);if(c<128)put(c);else{c-=128;put(I[c>>7]);put(c&127)}}
e.remove();return o}
""",
    # 8 bits per char, page read as windows-1252; 0x01 escapes NUL, CR, '<' and itself
    "latin1": r"""function eaglerRead(id,len){
var e=document.getElementById(id),s=e.textContent,o=new Uint8Array(len),j=0,i=0,l=s.length,c,k,M={},I=[0,13,60,1],D=new TextDecoder("windows-1252");
for(k=128;k<160;k++)M[D.decode(new Uint8Array([k])).charCodeAt(0)]=k;
while(j<len){c=s.charCodeAt(i++);if(c>255)c=M[c];if(c==1)c=I[s.charCodeAt(i++)-48];o[j++]=c}
e.remove();return o}
""",
    # bundle mode: the stub puts the payloads in window.__eagB as byte arrays
    "raw": r"""function eaglerRead(id,len){var o=window.__eagB[id];delete window.__eagB[id];return o}
""",
    # 15 bits per UTF-16 unit (page saved as UTF-16 with a BOM)
    "utf16": r"""function eaglerRead(id,len){
var e=document.getElementById(id),s=e.textContent,o=new Uint8Array(len),j=0,a=0,n=0,i=0,l=s.length;
for(;i<l&&j<len;i++){a=(a<<15)|(s.charCodeAt(i)-256);n+=15;while(n>=8){n-=8;if(j<len)o[j++]=(a>>n)&255}a&=(1<<n)-1}
e.remove();return o}
""",
}


def build_decoder(encoding):
    return ("<script>\n" + DEC_LZMA + DEC_CORE + DEC_READ[encoding] + "</script>\n").encode()


# ------------------------------------------------------------ html layer ----
PAY = re.compile(rb"\x00EAGPAY(\d+)\x00")
ENCODINGS = ("utf8", "utf16", "latin1", "raw")


class Pack:
    """collects the payload containers; the html gets a placeholder until the final encoding is known"""
    def __init__(self):
        self.items = []

    def add(self, id_, cont):
        self.items.append((id_, cont))
        return b"\x00EAGPAY%d\x00" % (len(self.items) - 1)


def process_blob(raw, name, o, log, sub):
    """audio shrink + stream packing. returns the container bytes"""
    if raw[:8] == EPK_MAGIC:
        cont = pack_epk(raw, o, log, sub)
    else:
        cont = pack_raw(raw, "bin", o)
    log("%s: %.2f MB (base64) -> %.2f MB (binary container)" % (name, len(raw) * 4 / 3 / 1e6, len(cont) / 1e6))
    return cont


# layout A: window.eaglercraftXOpts.assetsURI = "data:application/octet-stream;base64,....."
def pack_datauri(html, o, log, prog, pk):
    pat = re.compile(rb"data:application/octet-stream;base64,[A-Za-z0-9+/=]{1000,}")
    found = list(pat.finditer(html))
    if not found:
        return None
    res = []
    for k, m in enumerate(found):
        assert html[m.start() - 1:m.start()] in (b'"', b"'"), "data URI isn't a plain string literal"
        raw = base64.b64decode(m.group(0).split(b",", 1)[1])
        res.append(process_blob(raw, "data uri %d" % k, o, log, lambda f, k=k: prog((k + f) / len(found))))
        prog((k + 1) / len(found))
    scripts = [html.rfind(b"<script", 0, m.start()) for m in found]  # script tag holding each literal
    out, pos = bytearray(), 0
    for k, m in enumerate(found):
        if scripts[k] >= pos:  # first literal of a script tag: the text blocks go in front of it
            out += html[pos:scripts[k]]
            for j in range(len(found)):
                if scripts[j] == scripts[k]:
                    out += pk.add(b"eagler-blob-%d" % j, res[j])
            pos = scripts[k]
        out += html[pos:m.start() - 1] + b'eaglerEPK("eagler-blob-%d",%d)' % (k, len(res[k]))
        pos = m.end() + 1
    return bytes(out + html[pos:])


# layout B: var files = { "name": ["b64 chunk", ...] | "b64", ... };  read by decodeBase64ValueToU8()
FILES_ENTRY = re.compile(rb'\s*"([^"]+)"\s*:\s*(\[(?:\s*"[A-Za-z0-9+/=]*"\s*,?)*\s*\]|"[A-Za-z0-9+/=]*")\s*(,?)', re.S)
DECODE_FN = b"function decodeBase64ValueToU8(v){"


def pack_files_table(html, o, log, prog, pk):
    i = html.find(b"var files = {")
    if i < 0 or html.count(DECODE_FN) != 1:
        return None
    p = start = i + len(b"var files = {")
    entries = []
    while True:
        m = FILES_ENTRY.match(html, p)
        if not m:
            break
        entries.append(m); p = m.end()
    tail = re.compile(rb"\s*\}").match(html, p)
    if not entries or not tail:
        return None
    sc = html.rfind(b"<script", 0, i)
    scripts, items = bytearray(), []
    for k, m in enumerate(entries):
        name = m.group(1).decode()
        raw = b"".join(base64.b64decode(c) for c in re.findall(rb'"([A-Za-z0-9+/=]*)"', m.group(2)))
        cont = process_blob(raw, name, o, log, lambda f, k=k: prog((k + f) / len(entries)))
        prog((k + 1) / len(entries))
        scripts += pk.add(b"eagler-blob-%d" % k, cont)
        items.append(b'"%s": {__eag:"eagler-blob-%d",len:%d}' % (m.group(1), k, len(cont)))
    out = html[:sc] + scripts + html[sc:start] + b"\r\n\t\t\t\t" + b",\r\n\t\t\t\t".join(items) + b"\r\n\t\t\t" + html[tail.end() - 1:]
    return out.replace(DECODE_FN, DECODE_FN + b"if(v&&v.__eag)return eaglerDecode(v.__eag,v.len);", 1)


SCRIPT_RE = re.compile(rb"<script([^>]*)>(.*?)</script>", re.S | re.I)


def pack_inline_js(html, o, log, pk):
    """big inline <script> bodies (the game's own JS) -> LZMA container + tiny loader. returns (html, count)"""
    if re.search(rb"content-security-policy", html, re.I):
        log("inline scripts left as-is: page has a Content-Security-Policy")
        return html, 0
    out, pos, k = bytearray(), 0, 0
    for m in SCRIPT_RE.finditer(html):
        attrs, body = m.group(1), m.group(2)
        if len(body) < 100000 or re.search(rb"\bsrc\s*=", attrs):
            continue
        t = re.search(rb"type\s*=\s*[\"']?([^\"' >]+)", attrs, re.I)
        if t and t.group(1).lower() not in (b"text/javascript", b"application/javascript"):
            continue
        cont = pack_raw(body, "js", o)
        if len(cont) > 0.8 * len(body):
            continue
        log("inline script %d: %.2f MB -> %.2f MB (LZMA)" % (k, len(body) / 1e6, len(cont) / 1e6))
        out += html[pos:m.start()] + pk.add(b"eagler-js-%d" % k, cont)
        out += b'<script>eaglerRunJS("eagler-js-%d",%d)</script>' % (k, len(cont))
        pos = m.end(); k += 1
    return (bytes(out + html[pos:]) if k else html), k


def _ascii_shell(b):
    """make the non-payload html pure ASCII (needed when the page is read as windows-1252)"""
    if not re.search(rb"[\x80-\xff]", b):
        return b
    s = b.decode("utf-8")  # raises if the page isn't valid UTF-8
    def esc(m, js):
        return "".join(("\\u%04x" % ord(c) if js else "&#%d;" % ord(c)) if ord(c) < 0x10000 else
                       ("".join("\\u%04x" % u for u in struct.unpack("<2H", c.encode("utf-16-le"))) if js else "&#%d;" % ord(c))
                       for c in m.group(0))
    out, pos = [], 0
    for m in re.finditer(r"(<script[^>]*>)(.*?)(</script>)", s, re.S | re.I):
        out.append(re.sub(r"[^\x00-\x7f]+", lambda x: esc(x, False), s[pos:m.start()]))
        out.append(m.group(1) + re.sub(r"[^\x00-\x7f]+", lambda x: esc(x, True), m.group(2)) + m.group(3))
        pos = m.end()
    out.append(re.sub(r"[^\x00-\x7f]+", lambda x: esc(x, False), s[pos:]))
    return "".join(out).encode("ascii")


def finalize(html, pk, o, log):
    enc = o.encoding
    if enc == "raw":  # bundle mode: payloads leave the html and travel as binary inside the gzip
        o.blobs = [(i, c) for i, c in pk.items]
        return PAY.sub(b"", html)
    def elem(k, packed_bytes):
        return b'<script type="text/plain" id="' + pk.items[k][0] + b'">' + packed_bytes + b"</script>\n"
    if enc == "utf8":
        if not re.search(rb"charset\s*=\s*[\"']?utf-8", html[:4096], re.I):
            log("warning: page isn't declared UTF-8 near the top - add <meta charset=\"UTF-8\"> or the text blocks will break")
        return PAY.sub(lambda m: elem(int(m.group(1)), utf8_pack(pk.items[int(m.group(1))][1])), html)
    if enc == "latin1":
        parts = PAY.split(html)  # text, index, text, index, ...
        shell = b"".join(_ascii_shell(parts[i]) if i % 2 == 0 else b"\x00EAGPAY%s\x00" % parts[i] for i in range(len(parts)))
        shell = re.sub(rb"<meta\s+charset\s*=[^>]*>\s*", b"", shell, flags=re.I)
        shell = re.sub(rb"<meta\s+http-equiv\s*=\s*[\"']?content-type[\"']?[^>]*>\s*", b"", shell, flags=re.I)
        meta = b'<meta charset="windows-1252">\n'
        m = re.match(rb"\s*<!doctype[^>]*>\s*", shell, re.I)
        shell = (shell[:m.end()] + meta + shell[m.end():]) if m else meta + shell
        return PAY.sub(lambda m: elem(int(m.group(1)), latin1_pack(pk.items[int(m.group(1))][1])), shell)
    # utf16
    parts = PAY.split(html)
    out = [b"\xff\xfe"]
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(part.decode("utf-8").encode("utf-16-le"))
        else:
            k = int(part)
            out.append(('<script type="text/plain" id="%s">' % pk.items[k][0].decode()).encode("utf-16-le"))
            out.append(utf16_pack(pk.items[k][1]))
            out.append("</script>\n".encode("utf-16-le"))
    return b"".join(out)


# layout C: <script type="application/octet-stream" id="eag-inline-*" data-size="N"> base64 lines </script>
# the page's own loader decodes these, so they are rewritten in the same format (no page changes needed)
INLINE_BLK = re.compile(rb'(<script type="application/octet-stream" id="eag-inline-[^"]+" data-size=")(\d+)(">)(.*?)(</script>)', re.S)


def pack_inline_blocks(html, o, log, prog):
    found = [m for m in INLINE_BLK.finditer(html)]
    if not found:
        return None
    out, pos, saved = bytearray(), 0, 0
    for k, m in enumerate(found):
        body = m.group(4)
        raw = base64.b64decode(body)
        name = m.group(0)[m.group(0).index(b'id="') + 4:].split(b'"')[0].decode()
        if raw[:8] != EPK_MAGIC:
            log("%s: not an EPK, left as-is" % name)
            continue
        new = rebuild_epk(raw, o, log, lambda f, k=k: prog((k + f) / len(found)))
        if len(new) >= len(raw):
            log("%s: no gain, left as-is" % name)
            continue
        b64 = base64.b64encode(new)
        lines = [b64[i:i + 262144] for i in range(0, len(b64), 262144)]
        out += html[pos:m.start()] + m.group(1) + b"%d" % len(new) + m.group(3) + b"\n" + b"\n".join(lines) + b"\n" + m.group(5)
        pos = m.end()
        saved += len(raw) - len(new)
        log("%s: %.2f MB -> %.2f MB" % (name, len(raw) / 1e6, len(new) / 1e6))
        prog((k + 1) / len(found))
    if not saved:
        return None
    return bytes(out + html[pos:])


def html_pack(html, o, log, prog):
    if o.encoding not in ENCODINGS:
        raise ValueError("unknown text encoding %r" % o.encoding)
    res = pack_inline_blocks(html, o, log, prog)
    if res is not None:
        return res
    pk = Pack()
    o.blobs = []
    res = pack_datauri(html, o, log, prog, pk)
    if res is None:
        res = pack_files_table(html, o, log, prog, pk)
    res2, n = pack_inline_js(res if res is not None else html, o, log, pk)
    if n:
        res = res2
    if res is None:
        raise ValueError("couldn't find anything to shrink in this html (looked for base64 data URIs, a `var files = {...}` table and big inline scripts)")
    m = re.search(rb"<script[\s>]", res)  # decoder goes in front of the first script
    base = res
    res = base[:m.start()] + build_decoder(o.encoding) + base[m.start():]
    out = finalize(res, pk, o, log)
    if o.encoding == "raw" and o.blobs:  # also build the plain latin1 page: gzip can't shrink LZMA'd payloads, so it may win
        o.encoding = "latin1"
        try:
            o.plain = finalize(base[:m.start()] + build_decoder("latin1") + base[m.start():], pk, o, lambda *_: None)
        finally:
            o.encoding = "raw"
    return out


# ------------------------------------------------------------- one file -----
BUNDLE_STUB = r"""<script>(async function(){var e=document.getElementById("eagler-bundle"),s=e.textContent,o=new Uint8Array(%d),j=0,i=0,c,k,M={},I=[0,13,60,1],D=new TextDecoder("windows-1252"),B=%s,off=%d,u,h;
for(k=128;k<160;k++)M[D.decode(new Uint8Array([k])).charCodeAt(0)]=k;
while(j<o.length){c=s.charCodeAt(i++);if(c>255)c=M[c];if(c==1)c=I[s.charCodeAt(i++)-48];o[j++]=c}
e.remove();s="";
u=new Uint8Array(await new Response(new Blob([o]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());o=null;
window.__eagB={};h=new TextDecoder().decode(u.subarray(0,off));
for(k=0;k<B.length;k++){window.__eagB[B[k][0]]=u.subarray(off,off+B[k][1]);off+=B[k][1]}
document.open();document.write(h);document.close()})()</script>
"""


def bundle_page(html, blobs, o, log):
    """gzip the finished page plus its binary payloads, write that as 8-bit text (windows-1252) and put a
    tiny loader in front. The loader gunzips it with DecompressionStream, hands the payloads to the page
    as byte arrays (no text encoding overhead) and document.write()s the html."""
    if html[:2] in (b"\xff\xfe", b"\xfe\xff"):
        raise ValueError("bundle mode can't be combined with utf16 output")
    try:
        html.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("bundle mode needs a UTF-8 page")
    blob = html + b"".join(c for _, c in blobs)
    c = zlib.compressobj(9, zlib.DEFLATED, 31)
    gz = c.compress(blob) + c.flush()
    log("bundle: %.2f MB page + %.2f MB payloads -> %.2f MB gzip" % (len(html) / 1e6, (len(blob) - len(html)) / 1e6, len(gz) / 1e6))
    lst = "[" + ",".join('["%s",%d]' % (i.decode(), len(c_)) for i, c_ in blobs) + "]"
    return (b'<!DOCTYPE html>\n<meta charset="windows-1252">\n<script type="text/plain" id="eagler-bundle">' + latin1_pack(gz)
            + b"</script>\n" + (BUNDLE_STUB % (len(gz), lst, len(html))).encode() + b"<body style=margin:0;background:#000>\n")


def process_file(src, dst, o, log=print, prog=lambda f: None):
    log("== %s" % os.path.basename(src))
    data = open(src, "rb").read()
    if data[:8] == EPK_MAGIC:
        res = rebuild_epk(data, o, log, lambda f: prog(f))
    else:
        enc0, o.plain, o.blobs = o.encoding, None, []
        if o.bundle:
            o.encoding = "raw"
        res = html_pack(data, o, log, prog)
        if o.bundle:
            bundled = bundle_page(res, o.blobs, o, log)
            if o.plain is not None and len(o.plain) <= len(bundled):
                log("bundle: gzip can't shrink the already-LZMA'd payloads, using plain latin1 (%.2f MB) instead" % (len(o.plain) / 1e6))
                bundled = o.plain
            res = bundled
            o.encoding = enc0
    if o.cancel.is_set():
        raise Cancelled()
    if len(res) >= len(data):
        log("-> nothing saved (%.2f MB -> %.2f MB), no file written\n" % (len(data) / 1e6, len(res) / 1e6))
        return len(data), len(data)
    with open(dst, "wb") as f:
        f.write(res)
    prog(1.0)
    log("-> %s: %.2f MB -> %.2f MB (%.0f%% smaller)\n" % (os.path.basename(dst), len(data) / 1e6, len(res) / 1e6, 100 - 100 * len(res) / len(data)))
    return len(data), len(res)


def out_path(src, suffix, outdir):
    root, ext = os.path.splitext(os.path.basename(src))
    return os.path.join(outdir or os.path.dirname(os.path.abspath(src)), root + suffix + ext)


# -------------------------------------------------------------------- GUI ---
PRESETS = {
    "Recommended (24 kbps, UTF-8)": dict(kbps="24", rate="Keep original", mono=False, deflate=True, level=9, audio=True, enc="UTF-8 (works everywhere)"),
    "Smaller (24 kbps, UTF-16)": dict(kbps="24", rate="Keep original", mono=False, deflate=True, level=9, audio=True, enc="UTF-16 (about 7% smaller)"),
    "Smallest (16 kbps, 22 kHz cap, Latin-1)": dict(kbps="16", rate="22050 Hz", mono=False, deflate=True, level=9, audio=True, enc="Latin-1 (smallest, opened as a file)"),
    "No audio (remove all sounds)": dict(kbps="24", rate="Keep original", mono=False, deflate=True, level=9, audio=False, strip=True, enc="UTF-8 (works everywhere)"),
    "Fast (no audio re-encode)": dict(kbps="24", rate="Keep original", mono=False, deflate=True, level=4, audio=False, enc="UTF-8 (works everywhere)"),
}
ENC_LABELS = {"UTF-8 (works everywhere)": "utf8", "UTF-16 (about 7% smaller)": "utf16", "Latin-1 (smallest, opened as a file)": "latin1"}
RATES = {"Keep original": 0, "44100 Hz": 44100, "32000 Hz": 32000, "22050 Hz": 22050, "16000 Hz": 16000}


def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("MinifyEag")
    root.geometry("800x760")
    root.minsize(720, 660)
    try:
        ttk.Style().theme_use("vista" if sys.platform == "win32" else "clam")
    except tk.TclError:
        pass
    pad = dict(padx=8, pady=4)

    v_audio = tk.BooleanVar(value=True); v_strip = tk.BooleanVar(value=False); v_bundle = tk.BooleanVar(value=False); v_drop = tk.StringVar(value=""); v_langs = tk.StringVar(value=""); v_kbps = tk.StringVar(value="24"); v_rate = tk.StringVar(value="Keep original")
    v_mono = tk.BooleanVar(value=False); v_deflate = tk.BooleanVar(value=True); v_level = tk.IntVar(value=9)
    v_suffix = tk.StringVar(value=".small"); v_outdir = tk.StringVar(value=""); v_ffmpeg = tk.StringVar(value=find_ffmpeg())
    v_jobs = tk.IntVar(value=os.cpu_count() or 4); v_preset = tk.StringVar(value="Recommended (24 kbps, UTF-8)"); v_enc = tk.StringVar(value="UTF-8 (works everywhere)")
    v_ffstat = tk.StringVar(); v_status = tk.StringVar(value="Add one or more .html / .epk files to begin")
    files = []
    q = queue.Queue()
    state = {"opts": None, "running": False}

    # --- files
    fr = ttk.LabelFrame(root, text="Files"); fr.pack(fill="x", **pad)
    lb = tk.Listbox(fr, height=6, selectmode="extended", activestyle="none")
    lb.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
    sb = ttk.Scrollbar(fr, command=lb.yview); sb.pack(side="left", fill="y", pady=8); lb.config(yscrollcommand=sb.set)
    bf = ttk.Frame(fr); bf.pack(side="left", padx=8, pady=8)

    def add():
        for f in filedialog.askopenfilenames(title="Select Eaglercraft files", filetypes=[("Eaglercraft", "*.html *.htm *.epk"), ("All files", "*.*")]):
            if f not in files:
                files.append(f); lb.insert("end", f)
        v_status.set("%d file(s) selected" % len(files))
    def remove():
        for i in reversed(lb.curselection()):
            lb.delete(i); files.pop(i)
        v_status.set("%d file(s) selected" % len(files))
    def clear():
        lb.delete(0, "end"); files.clear(); v_status.set("Add one or more .html / .epk files to begin")
    ttk.Button(bf, text="Add files...", command=add).pack(fill="x", pady=2)
    ttk.Button(bf, text="Remove selected", command=remove).pack(fill="x", pady=2)
    ttk.Button(bf, text="Clear", command=clear).pack(fill="x", pady=2)

    # --- presets
    pf = ttk.Frame(root); pf.pack(fill="x", **pad)
    ttk.Label(pf, text="Preset:").pack(side="left")
    pc = ttk.Combobox(pf, textvariable=v_preset, values=list(PRESETS), state="readonly", width=40); pc.pack(side="left", padx=6)

    def apply_preset(*_):
        p = PRESETS[v_preset.get()]
        v_kbps.set(p["kbps"]); v_rate.set(p["rate"]); v_mono.set(p["mono"]); v_deflate.set(p["deflate"]); v_level.set(p["level"]); v_audio.set(p["audio"]); v_strip.set(p.get("strip", False)); v_enc.set(p["enc"])
    pc.bind("<<ComboboxSelected>>", apply_preset)

    # --- audio
    af = ttk.LabelFrame(root, text="Audio"); af.pack(fill="x", **pad)
    ttk.Checkbutton(af, text="Re-encode audio (.ogg inside EPK)", variable=v_audio).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
    ttk.Label(af, text="Bitrate (kbps):").grid(row=1, column=0, sticky="w", **pad)
    ttk.Combobox(af, textvariable=v_kbps, values=["8", "12", "16", "24", "32", "48", "64", "96", "128"], width=8).grid(row=1, column=1, sticky="w", **pad)
    ttk.Label(af, text="24 kbps recommended - good quality, big saving", foreground="#2a7a2a").grid(row=1, column=2, sticky="w", **pad)
    ttk.Label(af, text="Max sample rate:").grid(row=2, column=0, sticky="w", **pad)
    ttk.Combobox(af, textvariable=v_rate, values=list(RATES), state="readonly", width=14).grid(row=2, column=1, sticky="w", **pad)
    ttk.Label(af, text="Lowers rate only if the sound is higher. Keep original is safest.", foreground="#666").grid(row=2, column=2, sticky="w", **pad)
    ttk.Checkbutton(af, text="Force mono", variable=v_mono).grid(row=3, column=0, columnspan=2, sticky="w", **pad)
    ttk.Checkbutton(af, text="Remove all audio (no sound, much smaller)", variable=v_strip).grid(row=3, column=2, sticky="w", **pad)
    ttk.Label(af, text="ffmpeg:").grid(row=4, column=0, sticky="w", **pad)
    ttk.Entry(af, textvariable=v_ffmpeg, width=44).grid(row=4, column=1, columnspan=2, sticky="we", **pad)

    def ff_check(*_):
        ok, msg = ffmpeg_status(v_ffmpeg.get().strip() or "ffmpeg")
        v_ffstat.set(("OK: " if ok else "!! ") + msg + ("" if ok else " (audio will be skipped; browse to ffmpeg.exe)"))
        ff_lbl.config(foreground="#2a7a2a" if ok else "#b00020")
    def ff_browse():
        f = filedialog.askopenfilename(title="Locate ffmpeg", filetypes=[("ffmpeg", "ffmpeg*"), ("All files", "*.*")])
        if f:
            v_ffmpeg.set(f); ff_check()
    ttk.Button(af, text="Browse...", command=ff_browse).grid(row=4, column=3, **pad)
    ff_lbl = ttk.Label(af, textvariable=v_ffstat); ff_lbl.grid(row=5, column=0, columnspan=4, sticky="w", **pad)
    af.columnconfigure(2, weight=1)

    # --- compression
    cf = ttk.LabelFrame(root, text="Compression"); cf.pack(fill="x", **pad)
    ttk.Checkbutton(cf, text="Strong compression: LZMA for the game JS and assets, PNGs stored as raw pixels (unpacked by the page at load)", variable=v_deflate).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
    ttk.Checkbutton(cf, text="Bundle: gzip the whole page into one latin-1 text block (smallest; open the file directly)", variable=v_bundle).grid(row=5, column=0, columnspan=3, sticky="w", **pad)
    ttk.Label(cf, text="Keep only languages:").grid(row=6, column=0, sticky="w", **pad)
    ttk.Entry(cf, textvariable=v_langs, width=20).grid(row=6, column=1, sticky="w", **pad)
    ttk.Label(cf, text="e.g. en_US  (empty = keep all; every .lang file removed saves space)", foreground="#666").grid(row=6, column=2, sticky="w", **pad)
    ttk.Label(cf, text="Remove files matching:").grid(row=7, column=0, sticky="w", **pad)
    ttk.Entry(cf, textvariable=v_drop).grid(row=7, column=1, columnspan=2, sticky="we", **pad)
    ttk.Label(cf, text="space separated patterns, e.g. assets/eagler/boot_menu/* relay_download.zip", foreground="#666").grid(row=8, column=0, columnspan=3, sticky="w", **pad)
    ttk.Label(cf, text="Compression level:").grid(row=1, column=0, sticky="w", **pad)
    lvl_lbl = ttk.Label(cf, text="9", width=3)
    sc = ttk.Scale(cf, from_=1, to=9, orient="horizontal", command=lambda x: (v_level.set(int(round(float(x)))), lvl_lbl.config(text=str(v_level.get()))))
    sc.set(9); sc.grid(row=1, column=1, sticky="we", **pad); lvl_lbl.grid(row=1, column=2, **pad)
    ttk.Label(cf, text="9 = smallest (slower), 1 = fastest.", foreground="#666").grid(row=2, column=0, columnspan=3, sticky="w", **pad)
    ttk.Label(cf, text="Text encoding:").grid(row=3, column=0, sticky="w", **pad)
    ttk.Combobox(cf, textvariable=v_enc, values=list(ENC_LABELS), state="readonly", width=36).grid(row=3, column=1, columnspan=2, sticky="w", **pad)
    ttk.Label(cf, text="UTF-8 survives any server / fetch(). UTF-16 and Latin-1 only work when the browser reads the file itself (opened from disk or served as-is), not if a launcher re-reads it as text.", foreground="#666", wraplength=700, justify="left").grid(row=4, column=0, columnspan=3, sticky="w", **pad)
    cf.columnconfigure(1, weight=1)
    v_level.trace_add("write", lambda *_: (sc.set(v_level.get()), lvl_lbl.config(text=str(v_level.get()))))

    # --- output
    of = ttk.LabelFrame(root, text="Output"); of.pack(fill="x", **pad)
    ttk.Label(of, text="Name suffix:").grid(row=0, column=0, sticky="w", **pad)
    ttk.Entry(of, textvariable=v_suffix, width=12).grid(row=0, column=1, sticky="w", **pad)
    ttk.Label(of, text="Threads:").grid(row=0, column=2, sticky="e", **pad)
    ttk.Spinbox(of, from_=1, to=64, textvariable=v_jobs, width=5).grid(row=0, column=3, sticky="w", **pad)
    ttk.Label(of, text="Output folder:").grid(row=1, column=0, sticky="w", **pad)
    ttk.Entry(of, textvariable=v_outdir).grid(row=1, column=1, columnspan=3, sticky="we", **pad)
    ttk.Button(of, text="Browse...", command=lambda: v_outdir.set(filedialog.askdirectory() or v_outdir.get())).grid(row=1, column=4, **pad)
    ttk.Label(of, text="Leave the folder empty to save next to each original. Originals are never changed.", foreground="#666").grid(row=2, column=0, columnspan=5, sticky="w", **pad)
    of.columnconfigure(1, weight=1)

    # --- run
    rf = ttk.Frame(root); rf.pack(fill="x", **pad)
    start_btn = ttk.Button(rf, text="Start"); start_btn.pack(side="left")
    cancel_btn = ttk.Button(rf, text="Cancel", state="disabled"); cancel_btn.pack(side="left", padx=6)
    ttk.Label(rf, textvariable=v_status).pack(side="left", padx=10)
    bar = ttk.Progressbar(root, maximum=1000); bar.pack(fill="x", padx=8)
    lf = ttk.Frame(root); lf.pack(fill="both", expand=True, **pad)
    logbox = tk.Text(lf, height=8, state="disabled", wrap="word", font=("Consolas" if sys.platform == "win32" else "monospace", 9))
    logbox.pack(side="left", fill="both", expand=True)
    ls = ttk.Scrollbar(lf, command=logbox.yview); ls.pack(side="left", fill="y"); logbox.config(yscrollcommand=ls.set)

    def put_log(s):
        logbox.config(state="normal"); logbox.insert("end", s + "\n"); logbox.see("end"); logbox.config(state="disabled")

    def worker(jobs, o, suffix, outdir):
        tot_in = tot_out = 0
        try:
            for n, src in enumerate(jobs):
                if o.cancel.is_set():
                    raise Cancelled()
                q.put(("status", "File %d of %d: %s" % (n + 1, len(jobs), os.path.basename(src))))
                a, b = process_file(src, out_path(src, suffix, outdir), o,
                                    log=lambda s: q.put(("log", s)),
                                    prog=lambda f, n=n: q.put(("prog", (n + f) / len(jobs))))
                tot_in += a; tot_out += b
            q.put(("done", "Done: %.2f MB -> %.2f MB (%.0f%% smaller)" % (tot_in / 1e6, tot_out / 1e6, 100 - 100 * tot_out / max(tot_in, 1))))
        except Cancelled:
            q.put(("done", "Cancelled"))
        except Exception as e:
            q.put(("log", "ERROR: %s" % e)); q.put(("done", "Failed - see log"))

    def start():
        if state["running"]:
            return
        if not files:
            messagebox.showinfo("MinifyEag", "Add at least one file first."); return
        try:
            kbps = int(v_kbps.get()); assert 6 <= kbps <= 256
        except Exception:
            messagebox.showerror("MinifyEag", "Bitrate must be a number between 6 and 256."); return
        o = Opts(audio=v_audio.get(), strip_audio=v_strip.get(), bundle=v_bundle.get(), drop=v_drop.get().split(), langs=v_langs.get().strip(), kbps=kbps, max_rate=RATES.get(v_rate.get(), 0), mono=v_mono.get(), deflate=v_deflate.get(),
                 level=v_level.get(), jobs=max(1, v_jobs.get()), ffmpeg=v_ffmpeg.get().strip() or "ffmpeg", encoding=ENC_LABELS[v_enc.get()])
        suffix = v_suffix.get() or ".small"
        outdir = v_outdir.get().strip()
        if outdir and not os.path.isdir(outdir):
            messagebox.showerror("MinifyEag", "Output folder doesn't exist."); return
        state.update(opts=o, running=True)
        start_btn.config(state="disabled"); cancel_btn.config(state="normal"); bar["value"] = 0
        threading.Thread(target=worker, args=(list(files), o, suffix, outdir), daemon=True).start()
    def cancel():
        if state["opts"]:
            state["opts"].cancel.set(); v_status.set("Cancelling...")
    start_btn.config(command=start); cancel_btn.config(command=cancel)

    def poll():
        try:
            while True:
                k, v = q.get_nowait()
                if k == "log": put_log(v)
                elif k == "status": v_status.set(v)
                elif k == "prog": bar["value"] = v * 1000
                elif k == "done":
                    v_status.set(v); put_log(v); state["running"] = False
                    start_btn.config(state="normal"); cancel_btn.config(state="disabled")
                    if v.startswith("Done"): bar["value"] = 1000
        except queue.Empty:
            pass
        root.after(100, poll)
    poll(); ff_check()
    root.api = dict(files=files, start=start, vars=dict(kbps=v_kbps, outdir=v_outdir), lb=lb, state=state, v_status=v_status)
    return root


# -------------------------------------------------------------------- CLI ---
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="files to shrink (omit to open the window)")
    ap.add_argument("-o", "--output", help="output file (single input only)")
    ap.add_argument("-k", "--kbps", type=int, default=RECOMMENDED_KBPS, help="audio bitrate (default 24)")
    ap.add_argument("--max-rate", type=int, default=0, help="cap audio sample rate, e.g. 22050 (default keep)")
    ap.add_argument("--mono", action="store_true", help="force mono audio")
    ap.add_argument("--no-audio", action="store_true", help="skip audio re-encoding")
    ap.add_argument("--strip-audio", action="store_true", help="remove all sounds (.ogg) from the client")
    ap.add_argument("--drop", action="append", default=[], metavar="GLOB", help="remove EPK entries matching this pattern, e.g. 'assets/eagler/boot_menu/*' (repeatable)")
    ap.add_argument("--langs", default="", help="keep only these language files, e.g. en_US (comma separated)")
    ap.add_argument("--bundle", action="store_true", help="gzip the whole page and store it as latin1 text (smallest; open it as a file)")
    ap.add_argument("--no-deflate", action="store_true", help="store big assets uncompressed (no LZMA)")
    ap.add_argument("--encoding", choices=ENCODINGS, default="utf8",
                    help="how the data is written into the html: utf8 (safe everywhere, default), utf16 (~7%% smaller, page is saved as UTF-16) or latin1 (smallest, ~1.6%% overhead; only works when the browser reads the file as windows-1252, e.g. opened from disk)")
    ap.add_argument("-l", "--level", type=int, default=9, choices=range(1, 10), help="compression level 1-9 (default 9)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ffmpeg", default=None, help="path to ffmpeg")
    ap.add_argument("--suffix", default=".small")
    ap.add_argument("--gui", action="store_true")
    a = ap.parse_args()
    if a.gui or not a.inputs:
        try:
            gui().mainloop()
        except ImportError:
            sys.exit("the window needs tkinter (on Linux: sudo apt install python3-tk). Or pass files on the command line.")
        return
    o = Opts(audio=not a.no_audio, kbps=a.kbps, max_rate=a.max_rate, mono=a.mono, deflate=not a.no_deflate,
             level=a.level, strip_audio=a.strip_audio, drop=a.drop, langs=a.langs, bundle=a.bundle, jobs=a.jobs, ffmpeg=a.ffmpeg or find_ffmpeg(), encoding=a.encoding)
    for src in a.inputs:
        if a.output and len(a.inputs) == 1:
            dst = a.output
        else:
            dst = out_path(src, a.suffix, None)
        try:
            process_file(src, dst, o)
        except Exception as e:
            print("ERROR %s: %s" % (src, e)); sys.exit(1)


if __name__ == "__main__":
    main()
