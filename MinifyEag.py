#!/usr/bin/env python3
"""
eagler_shrink.py - shrink any Eaglercraft client file.

  python3 eagler_shrink.py client.html            -> client.small.html
  python3 eagler_shrink.py assets.epk             -> assets.small.epk
  python3 eagler_shrink.py client.html -k 24 -j 8

1. every .ogg inside the EPK is re-encoded to Vorbis at 24 kbps (kept as-is if the
   new file wouldn't be smaller)
2. for .html files, the base64 assets (+33% size) are replaced by base-122 style
   UTF-8 text blocks (+14% size), decoded at load time. handles a base64 assetsURI
   data URI and the `var files = {...}` chunk table layout. WASM-GC .epw blobs get
   the repack only (audio untouched)

needs: python3, numpy, ffmpeg (with libvorbis) on PATH
"""
import argparse, os, re, struct, subprocess, sys, zlib, base64
from concurrent.futures import ThreadPoolExecutor
import numpy as np

EPK_MAGIC = b"EAGPKG$$"
B64_MAGIC = "RUFHUEtHJCQ"  # base64 of "EAGPKG$$"
ILLEGAL = (0, 10, 13, 60)  # NUL, LF, CR, '<' -> not safe raw inside a <script> block


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


def epk_build(head, comp, ents, tail, footer):
    o = bytearray()
    for t, n, x in ents:
        nb = n.encode()
        o += t + bytes([len(nb)]) + nb
        if t == b"FILE":
            o += struct.pack(">II", len(x) + 5, zlib.crc32(x)) + x + b":"
        else:
            o += struct.pack(">I", len(x)) + x
        o += b">"
    o += b"END$" + tail
    if comp == b"G":
        c = zlib.compressobj(9, zlib.DEFLATED, 31); body = c.compress(bytes(o)) + c.flush()
    elif comp == b"Z":
        c = zlib.compressobj(9, zlib.DEFLATED, 15); body = c.compress(bytes(o)) + c.flush()
    else:
        body = bytes(o)
    return head + comp + body + footer


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


def to_vorbis(data, kbps):
    info = ogg_info(data)
    if info is None or info[2] <= kbps:  # unreadable, or already at/below target: leave alone
        return None
    sr, ch, _ = info  # keep original sample rate + channels
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-map_metadata", "-1",
         "-c:a", "libvorbis", "-b:a", "%dk" % kbps, "-ar", str(sr), "-ac", str(ch),
         "-f", "ogg", "pipe:1"],
        input=data, capture_output=True)
    return r.stdout if r.returncode == 0 and r.stdout else None


def shrink_audio(ents, kbps, jobs):
    idx = [i for i, e in enumerate(ents) if e[0] == b"FILE" and e[1].lower().endswith(".ogg")]
    with ThreadPoolExecutor(jobs) as ex:
        outs = list(ex.map(lambda i: to_vorbis(ents[i][2], kbps), idx))
    before = after = kept = failed = 0
    for i, new in zip(idx, outs):
        old = ents[i][2]; before += len(old)
        if new is None:
            kept += 1; after += len(old)
        elif len(new) < len(old):
            ents[i][2] = new; after += len(new)
        else:
            kept += 1; after += len(old)
    print("audio: %d ogg, %.2f MB -> %.2f MB (%d left as-is: already <= target or not smaller, %d failed)"
          % (len(idx), before / 1e6, after / 1e6, kept, failed))


# -------------------------------------------- base-122 style UTF-8 packing --
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
    buf = np.frombuffer(data + b"\0" * (-n % 7) + b"\xff" * 7, dtype=np.uint8).reshape(-1, 7).astype(np.uint8)
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


DECODER = b"""<script>
function eaglerDecode(id,len){
var e=document.getElementById(id),s=e.textContent,out=new Uint8Array(len),o=0,acc=0,n=0,I=[0,10,13,60];
function put(v){acc=((acc<<7)|v)&16383;n+=7;if(n>=8){n-=8;if(o<len)out[o++]=(acc>>n)&255;}}
for(var i=0,l=s.length;i<l;i++){var c=s.charCodeAt(i);if(c<128)put(c);else{c-=128;put(I[c>>7]);put(c&127);}}
e.remove();return out;}
function eaglerEPK(id,len){return URL.createObjectURL(new Blob([eaglerDecode(id,len)],{type:"application/octet-stream"}));}
</script>
"""


def process_blob(raw, name, kbps, jobs):
    """shrink audio if it's an EPK, then pack to UTF-8 text. returns (packed, decoded_len)"""
    if raw[:8] == EPK_MAGIC:
        head, comp, ents, tail, footer = epk_parse(raw)
        shrink_audio(ents, kbps, jobs)
        raw = epk_build(head, comp, ents, tail, footer)
    packed = utf8_pack(raw)
    print("%s: %.2f MB -> %.2f MB utf-8 text" % (name, len(raw) * 4 / 3 / 1e6, len(packed) / 1e6))
    return packed, len(raw)


def payload(k, packed):
    return b'<script type="text/plain" id="eagler-blob-%d">' % k + packed + b"</script>\n"


# layout A: window.eaglercraftXOpts.assetsURI = "data:application/octet-stream;base64,...."
def pack_datauri(html, kbps, jobs):
    pat = re.compile(rb"data:application/octet-stream;base64,[A-Za-z0-9+/=]{1000,}")
    found = list(pat.finditer(html))
    if not found:
        return None
    out, pos = bytearray(), 0
    for k, m in enumerate(found):
        quote = html[m.start() - 1:m.start()]
        assert quote in (b'"', b"'"), "data URI isn't a plain string literal"
        raw = base64.b64decode(m.group(0).split(b",", 1)[1])
        packed, n = process_blob(raw, "data uri %d" % k, kbps, jobs)
        sc = html.rfind(b"<script", 0, m.start())  # script tag holding the literal
        out += html[pos:sc]
        if k == 0:
            out += DECODER
        out += payload(k, packed)
        out += html[sc:m.start() - 1] + b'eaglerEPK("eagler-blob-%d",%d)' % (k, n)
        pos = m.end() + 1
    return bytes(out + html[pos:])


# layout B: var files = { "name": ["b64 chunk", ...] | "b64", ... };  read by decodeBase64ValueToU8()
FILES_ENTRY = re.compile(rb'\s*"([^"]+)"\s*:\s*(\[(?:\s*"[A-Za-z0-9+/=]*"\s*,?)*\s*\]|"[A-Za-z0-9+/=]*")\s*(,?)', re.S)
DECODE_FN = b"function decodeBase64ValueToU8(v){"


def pack_files_table(html, kbps, jobs):
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
        packed, n = process_blob(raw, name, kbps, jobs)
        scripts += payload(k, packed)
        items.append(b'"%s": {__eag:"eagler-blob-%d",len:%d}' % (m.group(1), k, n))
    out = html[:sc] + DECODER + scripts + html[sc:start] + b"\r\n\t\t\t\t" + b",\r\n\t\t\t\t".join(items) + b"\r\n\t\t\t" + html[tail.end() - 1:]
    return out.replace(DECODE_FN, DECODE_FN + b"if(v&&v.__eag)return eaglerDecode(v.__eag,v.len);", 1)


def html_pack(html, kbps, jobs):
    res = pack_datauri(html, kbps, jobs) or pack_files_table(html, kbps, jobs)
    if res is None:
        sys.exit("couldn't find embedded assets in this html (looked for a base64 data URI and a `var files = {...}` table)")
    if b"charset" not in html[:2048].lower():
        print("warning: no <meta charset> near top of file - add <meta charset=\"UTF-8\"> or the text blocks will break")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("-k", "--kbps", type=int, default=24)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    root, ext = os.path.splitext(a.input)
    outp = a.output or root + ".small" + ext
    data = open(a.input, "rb").read()
    if data[:8] == EPK_MAGIC:
        head, comp, ents, tail, footer = epk_parse(data)
        shrink_audio(ents, a.kbps, a.jobs)
        res = epk_build(head, comp, ents, tail, footer)
    else:
        res = html_pack(data, a.kbps, a.jobs)
    open(outp, "wb").write(res)
    print("%s: %.2f MB -> %s: %.2f MB" % (a.input, len(data) / 1e6, outp, len(res) / 1e6))


if __name__ == "__main__":
    main()
