"""Bounded, dependency-free output meters for a TRUSTED attested encoder.

These validate PCM bytes / MP4 container and video sample structure, not arbitrary
H.264 or AAC bitstream correctness, provenance, or billing authority. Full ffmpeg
video+audio decode remains a live acceptance criterion for the deployed encoder.
No subprocess, network, codec library, byte-rate estimate, or wall clock is used.

MP4 support is deliberately narrow: nonfragmented, self-contained avc1/H.264,
one video track, optional AAC audio, constant 24 fps, <=4096 video samples, and
native canvases <=3072 per axis and <=3072*768 pixels. Audio duration (including
AAC padding) never contributes to the result. A single full-length, unit-rate
edit may remove a validated composition-time offset for B-frame delay; trimming,
gaps, complex edits, rotations, external media, and unsupported tables fail closed.
Movie-tick rounding is checked only for consistency, never used as the meter.
Request-duration policy (Cosmos3-Super: at most 189 frames at 24 fps) and WAV
format consistency across streamed chunks belong to the caller.
"""

from __future__ import annotations

import base64
import binascii
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

MAX_WAV_BYTES = 4 * 1024 * 1024
MAX_MP4_BYTES = 128 * 1024 * 1024
MAX_VIDEO_SAMPLES = 4096
_MAX_BOXES = 16384
_MAX_DEPTH = 12
_MAX_TRACKS = 8
_MAX_MDAT = 64
_MATRIX = (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}
_FRAGMENTED = {b"moof", b"mvex", b"mfra", b"styp", b"sidx", b"traf"}


class MeterError(ValueError):
    """Fixed, non-content-bearing error for all invalid/unsupported input."""

    def __init__(self) -> None:
        super().__init__("invalid or unsupported media")


def _require(ok: bool) -> None:
    if not ok:
        raise MeterError


def wav_frames(encoded_base64: str) -> tuple[int, int, int, int]:
    """Return actual (PCM frames, sample rate, channels, bytes/sample).

    The decoded WAV, including headers, is limited to 4 MiB before parsing;
    encoded length is checked before allocating the decoded buffer. Accepts PCM
    integer WAV (including unambiguous PCM extensible), not float/compressed WAV.
    A partial final PCM frame, an empty data chunk, or any truncated chunk fails.
    """
    _require(isinstance(encoded_base64, str))
    _require(0 < len(encoded_base64) <= 4 * ((MAX_WAV_BYTES + 2) // 3))
    try:
        raw = base64.b64decode(encoded_base64, validate=True)
    except (ValueError, binascii.Error):
        raise MeterError from None
    _require(len(raw) <= MAX_WAV_BYTES)
    # Also reject noncanonical padding / ignored pad bits.
    _require(base64.b64encode(raw).decode("ascii") == encoded_base64)
    _require(len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE")
    _require(int.from_bytes(raw[4:8], "little") == len(raw) - 8)
    fmt = None
    pcm_size = None
    pos = 12
    chunks = 0
    while pos < len(raw):
        chunks += 1
        _require(chunks <= _MAX_BOXES and len(raw) - pos >= 8)
        kind = raw[pos : pos + 4]
        size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        start = pos + 8
        end = start + size
        _require(end <= len(raw))
        if kind == b"fmt ":
            _require(fmt is None and pcm_size is None)
            fmt = raw[start:end]
        elif kind == b"data":
            _require(fmt is not None and pcm_size is None)
            pcm_size = size
        # Python's wave writer omits padding for an odd terminal data chunk.
        # It is complete PCM, not a truncated sample, if the RIFF length agrees.
        pos = end + (size & 1)
        if end == len(raw) and kind == b"data":
            pos = end
        _require(pos <= len(raw))
    _require(fmt is not None and pcm_size is not None and pcm_size > 0)
    _require(len(fmt) >= 16)
    tag = int.from_bytes(fmt[:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    byte_rate = int.from_bytes(fmt[8:12], "little")
    align = int.from_bytes(fmt[12:14], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if tag == 0xFFFE:
        _require(len(fmt) == 40 and int.from_bytes(fmt[16:18], "little") == 22)
        _require(int.from_bytes(fmt[18:20], "little") == bits)
        _require(fmt[24:] == bytes.fromhex("0100000000001000800000aa00389b71"))
        mask = int.from_bytes(fmt[20:24], "little")
        _require(mask == 0 or mask.bit_count() == channels)
    else:
        _require(tag == 1 and (len(fmt) == 16 or (len(fmt) == 18 and fmt[16:] == b"\0\0")))
    _require(channels > 0 and rate > 0 and bits in (8, 16, 24, 32))
    width = bits // 8
    _require(align == channels * width and byte_rate == rate * align)
    _require(pcm_size % align == 0)
    return pcm_size // align, rate, channels, width


@dataclass(frozen=True)
class Mp4Metadata:
    """Container-validated video metadata, not a codec decode certificate."""

    duration: Fraction
    frames: int
    width: int
    height: int


@dataclass(frozen=True)
class _Box:
    kind: bytes
    start: int  # payload start (absolute file offset)
    end: int
    payload: memoryview
    children: tuple[_Box, ...] = ()


def _u(data: memoryview, offset: int, size: int = 4, *, signed: bool = False) -> int:
    _require(offset >= 0 and offset + size <= len(data))
    return int.from_bytes(data[offset : offset + size], "big", signed=signed)


class _Parser:
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.count = 0

    def boxes(self, start: int, end: int, depth: int = 0) -> tuple[_Box, ...]:
        _require(depth <= _MAX_DEPTH and 0 <= start <= end <= len(self.data))
        result = []
        while start < end:
            self.count += 1
            _require(self.count <= _MAX_BOXES and end - start >= 8)
            size = _u(self.data, start)
            kind = bytes(self.data[start + 4 : start + 8])
            header = 8
            if size == 1:
                _require(end - start >= 16)
                size = _u(self.data, start + 8, 8)
                header = 16
            elif size == 0:
                # ISO-BMFF size zero extends to EOF, not the parent boundary.
                _require(depth == 0)
                size = end - start
            _require(header <= size <= end - start and kind not in _FRAGMENTED)
            body = start + header
            stop = start + size
            children = ()
            if kind in _CONTAINERS:
                children = self.boxes(body, stop, depth + 1)
            elif kind == b"meta":
                _require(stop - body >= 4 and _u(self.data, body) == 0)
                children = self.boxes(body + 4, stop, depth + 1)
            result.append(_Box(kind, body, stop, self.data[body:stop], children))
            start = stop
        return tuple(result)


def _one(boxes: tuple[_Box, ...], kind: bytes, *, optional: bool = False) -> _Box | None:
    matches = [box for box in boxes if box.kind == kind]
    _require(len(matches) <= 1 and (optional or len(matches) == 1))
    return matches[0] if matches else None


def _allowed(boxes: tuple[_Box, ...], kinds: set[bytes]) -> None:
    _require(all(box.kind in kinds for box in boxes))
    # Only tracks and top-level media/free boxes may repeat at supported scopes.
    for kind in {box.kind for box in boxes} - {b"trak", b"mdat", b"free", b"skip"}:
        _one(boxes, kind)


def _full(box: _Box, *, versions: tuple[int, ...] = (0,), flags: int = 0):
    _require(len(box.payload) >= 4)
    version = box.payload[0]
    _require(version in versions and _u(box.payload, 1, 3) == flags)
    return version, box.payload[4:]


def _clock(box: _Box) -> tuple[int, int]:
    version, data = _full(box, versions=(0, 1))
    _require(len(data) == ({b"mvhd": (96, 108), b"mdhd": (20, 32)}[box.kind][version]))
    offset = 8 if version == 0 else 16
    scale = _u(data, offset)
    duration = _u(data, offset + 4, 4 if version == 0 else 8)
    _require(scale > 0 and 0 < duration < (1 << (32 if version == 0 else 64)) - 1)
    return scale, duration


def _track_header(box: _Box) -> tuple[int, int, int, int]:
    _require(len(box.payload) >= 4)
    flags = _u(box.payload, 1, 3)
    _require(flags & 3 == 3 and flags & ~7 == 0)
    version, data = _full(box, versions=(0, 1), flags=flags)
    _require(len(data) == (80 if version == 0 else 92))
    offset = 8 if version == 0 else 16
    track_id = _u(data, offset)
    duration = _u(data, offset + 8, 4 if version == 0 else 8)
    matrix = offset + (28 if version == 0 else 32)
    _require(tuple(_u(data, matrix + i * 4) for i in range(9)) == _MATRIX)
    width, height = _u(data, matrix + 36), _u(data, matrix + 40)
    _require(track_id > 0 and duration > 0 and width % 65536 == height % 65536 == 0)
    return track_id, duration, width // 65536, height // 65536


def _handler(box: _Box) -> bytes:
    _, data = _full(box)
    _require(len(data) >= 20)
    return bytes(data[4:8])


def _table(box: _Box, width: int, *, versions: tuple[int, ...] = (0,)):
    version, data = _full(box, versions=versions)
    count = _u(data, 0)
    _require(0 < count <= MAX_VIDEO_SAMPLES and len(data) == 4 + count * width)
    return version, data[4:], count


def _sample_sizes(box: _Box) -> list[int]:
    _, data = _full(box)
    size, count = _u(data, 0), _u(data, 4)
    _require(0 < count <= MAX_VIDEO_SAMPLES)
    if size:
        _require(len(data) == 8 and size <= MAX_MP4_BYTES)
        return [size] * count
    _require(len(data) == 8 + count * 4)
    sizes = [_u(data, 8 + i * 4) for i in range(count)]
    _require(all(0 < size <= MAX_MP4_BYTES for size in sizes))
    return sizes


def _sample_ticks(box: _Box, count: int, scale: int) -> list[int]:
    _, data, entries = _table(box, 8)
    ticks = []
    for i in range(entries):
        samples, delta = _u(data, i * 8), _u(data, i * 8 + 4)
        _require(0 < samples <= count - len(ticks) and delta > 0 and delta * 24 == scale)
        ticks.extend([delta] * samples)
    _require(len(ticks) == count)
    return ticks


def _composition_start(box: _Box | None, ticks: list[int]) -> int:
    offsets = [0] * len(ticks)
    if box is not None:
        version, data, entries = _table(box, 8, versions=(0, 1))
        offsets = []
        total = sum(ticks)
        for i in range(entries):
            samples = _u(data, i * 8)
            offset = _u(data, i * 8 + 4, signed=version == 1)
            _require(0 < samples <= len(ticks) - len(offsets) and abs(offset) <= total)
            offsets.extend([offset] * samples)
        _require(len(offsets) == len(ticks))
    # Full-length B-frame reordering is a permutation of contiguous presentation
    # intervals. A matching edit can shift those intervals, but cannot trim them.
    intervals = []
    dts = 0
    for delta, offset in zip(ticks, offsets, strict=True):
        intervals.append((dts + offset, dts + offset + delta))
        dts += delta
    intervals.sort()
    _require(intervals[0][0] >= 0)
    _require(all(a[1] == b[0] for a, b in pairwise(intervals)))
    return intervals[0][0]


def _edit(box: _Box | None, movie_ticks: int, composition_start: int) -> None:
    if box is None:
        _require(composition_start == 0)
        return
    _allowed(box.children, {b"elst"})
    elst = _one(box.children, b"elst")
    version, data = _full(elst, versions=(0, 1))
    _require(_u(data, 0) == 1 and len(data) == (16 if version == 0 else 24))
    width = 4 if version == 0 else 8
    duration = _u(data, 4, width)
    media_time = _u(data, 4 + width, width, signed=True)
    _require(duration == movie_ticks and media_time == composition_start)
    _require(_u(data, 4 + 2 * width) == 0x00010000)  # signed 16.16 rate == 1


def _avcc(box: _Box) -> None:
    # Validate the configuration envelope and bounded parameter-set lengths only,
    # not SPS/PPS codec semantics. A full decode is an independent live criterion.
    data = box.payload
    _require(len(data) >= 7 and data[0] == 1 and data[4] == 255 and data[5] & 224 == 224)
    pos = 6

    def parameter_sets(count: int) -> None:
        nonlocal pos
        for _ in range(count):
            size = _u(data, pos, 2)
            pos += 2
            _require(size > 0 and pos + size <= len(data))
            pos += size

    sps = data[5] & 31
    _require(sps > 0)
    parameter_sets(sps)
    _require(pos < len(data) and data[pos] > 0)
    pps = data[pos]
    pos += 1
    parameter_sets(pps)
    if pos < len(data):
        _require(data[1] in (100, 110, 122, 144, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135))
        _require(len(data) - pos >= 4)
        _require(
            data[pos] & 252 == 252 and data[pos + 1] & 248 == 248 and data[pos + 2] & 248 == 248
        )
        ext = data[pos + 3]
        pos += 4
        parameter_sets(ext)
    _require(pos == len(data))


def _descriptors(data: memoryview) -> dict[int, memoryview]:
    result = {}
    pos = 0
    while pos < len(data):
        tag = data[pos]
        pos += 1
        size = 0
        for _ in range(4):
            byte = _u(data, pos, 1)
            pos += 1
            size = (size << 7) | (byte & 127)
            if not byte & 128:
                break
        else:
            raise MeterError
        _require(tag not in result and pos + size <= len(data))
        result[tag] = data[pos : pos + size]
        pos += size
    return result


def _aac(box: _Box) -> None:
    _, data = _full(box)
    descriptors = _descriptors(data)
    _require(set(descriptors) == {3})
    es = descriptors[3]
    _require(len(es) >= 3 and es[2] == 0)  # no dependent streams, URL, or OCR
    descriptors = _descriptors(es[3:])
    _require(set(descriptors) == {4, 6} and bytes(descriptors[6]) == b"\x02")
    config = descriptors[4]
    _require(len(config) >= 13 and config[0] == 0x40 and config[1] == 0x15)
    descriptors = _descriptors(config[13:])
    _require(set(descriptors) == {5})
    asc = descriptors[5]
    _require(len(asc) >= 2 and asc[0] >> 3 == 2)  # AAC-LC, as emitted by ffmpeg


def _description(parser: _Parser, box: _Box, *, video: bool) -> tuple[int, int]:
    _, data = _full(box)
    _require(_u(data, 0) == 1)
    entries = parser.boxes(box.start + 8, box.end, 6)
    _require(len(entries) == 1)
    entry = entries[0]
    header = 78 if video else 28
    _require(entry.kind == (b"avc1" if video else b"mp4a") and len(entry.payload) >= header)
    _require(bytes(entry.payload[:6]) == bytes(6) and _u(entry.payload, 6, 2) == 1)
    extensions = parser.boxes(entry.start + header, entry.end, 7)
    if not video:
        # AAC audio sample bytes and duration are NOT the video meter. Restrict
        # the sample-entry form, but leave audio codec validation to live decode.
        _require(_u(entry.payload, 8, 2) == 0)
        _allowed(extensions, {b"esds", b"btrt"})
        _aac(_one(extensions, b"esds"))
        return 0, 0
    width, height = _u(entry.payload, 24, 2), _u(entry.payload, 26, 2)
    _require(0 < width <= 3072 and 0 < height <= 3072 and width * height <= 3072 * 768)
    _require(_u(entry.payload, 40, 2) == 1)
    _allowed(extensions, {b"avcC", b"pasp", b"btrt", b"colr"})
    _avcc(_one(extensions, b"avcC"))
    pasp = _one(extensions, b"pasp", optional=True)
    if pasp is not None:
        _require(len(pasp.payload) == 8 and _u(pasp.payload, 0) == _u(pasp.payload, 4) > 0)
    return width, height


def _data_reference(parser: _Parser, box: _Box) -> None:
    _allowed(box.children, {b"dref"})
    dref = _one(box.children, b"dref")
    _, data = _full(dref)
    _require(_u(data, 0) == 1)
    refs = parser.boxes(dref.start + 8, dref.end, 6)
    _require(len(refs) == 1 and refs[0].kind == b"url ")
    _, data = _full(refs[0], flags=1)
    _require(len(data) == 0)  # self-contained, no external URL/path


def _chunks(stbl: tuple[_Box, ...], sizes: list[int], mdats: list[tuple[int, int]]) -> None:
    offsets_box = _one(stbl, b"stco", optional=True)
    offsets64 = _one(stbl, b"co64", optional=True)
    _require((offsets_box is None) != (offsets64 is None))
    width = 8 if offsets64 is not None else 4
    _, offsets, chunks = _table(offsets64 if offsets64 is not None else offsets_box, width)
    _require(chunks <= len(sizes))
    _, layout, entries = _table(_one(stbl, b"stsc"), 12)
    runs = []
    for i in range(entries):
        first, samples, description = (_u(layout, i * 12 + j * 4) for j in range(3))
        _require(1 <= first <= chunks and 0 < samples <= len(sizes) and description == 1)
        _require((not runs and first == 1) or (bool(runs) and first > runs[-1][0]))
        runs.append((first, samples))
    intervals = []
    starts = [start for start, _ in mdats]
    sample = run = 0
    for chunk in range(1, chunks + 1):
        if run + 1 < len(runs) and chunk == runs[run + 1][0]:
            run += 1
        count = runs[run][1]
        _require(sample + count <= len(sizes))
        start = _u(offsets, (chunk - 1) * width, width)
        end = start + sum(sizes[sample : sample + count])
        sample += count
        mdat = bisect_right(starts, start) - 1
        _require(mdat >= 0 and mdats[mdat][0] <= start < end <= mdats[mdat][1])
        intervals.append((start, end))
    _require(sample == len(sizes))
    intervals.sort()
    _require(all(a[1] <= b[0] for a, b in pairwise(intervals)))


def _auxiliary_tables(stbl: tuple[_Box, ...], count: int) -> None:
    stss = _one(stbl, b"stss", optional=True)
    if stss is not None:
        _, data, entries = _table(stss, 4)
        previous = 0
        for i in range(entries):
            sample = _u(data, i * 4)
            _require(previous < sample <= count)
            previous = sample
    sdtp = _one(stbl, b"sdtp", optional=True)
    if sdtp is not None:
        _, data = _full(sdtp)
        _require(len(data) == count)


def mp4_metadata(data: bytes) -> Mp4Metadata:
    """Return exact video duration, sample count, and native dimensions.

    Limits are applied before parsing and before expanding tables. Video chunks
    must be nonoverlapping and wholly inside mdat payload(s); every sample must
    have positive byte size. Unused mdat bytes (e.g. AAC) are not counted. Unknown
    video tables fail closed. Bounded ancillary user metadata is not interpreted.
    """
    _require(isinstance(data, bytes) and 0 < len(data) <= MAX_MP4_BYTES)
    parser = _Parser(data)
    top = parser.boxes(0, len(data))
    _allowed(top, {b"ftyp", b"moov", b"mdat", b"free", b"skip"})
    ftyp = _one(top, b"ftyp")
    _require(len(ftyp.payload) >= 8 and len(ftyp.payload) % 4 == 0)
    _require(bytes(ftyp.payload[:4]) in {b"isom", b"iso2", b"mp41", b"mp42", b"avc1"})
    moov = _one(top, b"moov")
    mdats = [(box.start, box.end) for box in top if box.kind == b"mdat"]
    _require(0 < len(mdats) <= _MAX_MDAT)
    _allowed(moov.children, {b"mvhd", b"trak", b"udta", b"meta"})
    movie_scale, movie_duration = _clock(_one(moov.children, b"mvhd"))
    tracks = [box for box in moov.children if box.kind == b"trak"]
    _require(0 < len(tracks) <= _MAX_TRACKS)
    result = None
    track_ids = set()
    for track in tracks:
        _allowed(track.children, {b"tkhd", b"mdia", b"edts", b"udta", b"meta"})
        track_id, track_duration, width, height = _track_header(_one(track.children, b"tkhd"))
        _require(track_id not in track_ids)
        track_ids.add(track_id)
        mdia = _one(track.children, b"mdia")
        _allowed(mdia.children, {b"mdhd", b"hdlr", b"minf"})
        scale, duration = _clock(_one(mdia.children, b"mdhd"))
        handler = _handler(_one(mdia.children, b"hdlr"))
        _require(handler in (b"vide", b"soun"))
        minf = _one(mdia.children, b"minf")
        _allowed(minf.children, {b"vmhd" if handler == b"vide" else b"smhd", b"dinf", b"stbl"})
        media_header = _one(minf.children, b"vmhd" if handler == b"vide" else b"smhd")
        _, header_data = _full(media_header, flags=1 if handler == b"vide" else 0)
        _require(len(header_data) == (8 if handler == b"vide" else 4))
        _data_reference(parser, _one(minf.children, b"dinf"))
        stbl = _one(minf.children, b"stbl").children
        dimensions = _description(parser, _one(stbl, b"stsd"), video=handler == b"vide")
        if handler == b"soun":
            _require(width == height == 0)
            continue
        _require(result is None and dimensions == (width, height))
        _allowed(
            stbl, {b"stsd", b"stts", b"stsz", b"stsc", b"stco", b"co64", b"ctts", b"stss", b"sdtp"}
        )
        sizes = _sample_sizes(_one(stbl, b"stsz"))
        ticks = _sample_ticks(_one(stbl, b"stts"), len(sizes), scale)
        total = sum(ticks)
        _require(total == duration)
        exact_duration = Fraction(total, scale)
        movie_ticks = (total * movie_scale + scale - 1) // scale
        _require(track_duration == movie_ticks and movie_duration >= movie_ticks)
        composition_start = _composition_start(_one(stbl, b"ctts", optional=True), ticks)
        _edit(_one(track.children, b"edts", optional=True), movie_ticks, composition_start)
        _chunks(stbl, sizes, mdats)
        _auxiliary_tables(stbl, len(sizes))
        result = Mp4Metadata(exact_duration, len(sizes), width, height)
    _require(result is not None)
    return result


def mp4_duration(data: bytes) -> Fraction:
    """Return sum(video stts sample durations) / video mdhd timescale, exactly."""
    return mp4_metadata(data).duration
