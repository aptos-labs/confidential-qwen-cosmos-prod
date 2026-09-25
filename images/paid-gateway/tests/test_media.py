"""Synthetic container fixtures: structural tests, intentionally NOT codec tests.

No ffmpeg, network, model runtime, or package-level gateway imports are needed.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import struct
import sys
import wave
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "paid_gateway" / "media.py"
_SPEC = importlib.util.spec_from_file_location("_standalone_media_meter", _PATH)
media = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = media
_SPEC.loader.exec_module(media)


def u32(*values):
    return struct.pack(">" + "I" * len(values), *values)


def box(kind, payload=b"", *, wide=False):
    if wide:
        return u32(1) + kind + struct.pack(">Q", 16 + len(payload)) + payload
    return u32(8 + len(payload)) + kind + payload


def full(payload=b"", *, version=0, flags=0):
    return bytes([version]) + flags.to_bytes(3, "big") + payload


def table(kind, rows, *, version=0, signed=False):
    values = b"".join(
        value.to_bytes(4, "big", signed=signed and i % 2 == 1)
        for row in rows
        for i, value in enumerate(row)
    )
    return box(kind, full(u32(len(rows)) + values, version=version))


def clock(kind, scale, duration, version=0):
    times = bytes(8 if version == 0 else 16)
    ticks = duration.to_bytes(4 if version == 0 else 8, "big")
    rest = bytes(4) if kind == b"mdhd" else u32(65536) + bytes(12) + matrix() + bytes(24) + u32(3)
    return box(kind, full(times + u32(scale) + ticks + rest, version=version))


def matrix():
    return u32(65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)


def tkhd(track_id, duration, width, height, version=0):
    data = bytes(8 if version == 0 else 16) + u32(track_id, 0)
    data += duration.to_bytes(4 if version == 0 else 8, "big")
    data += bytes(16) + matrix() + u32(width << 16, height << 16)
    return box(b"tkhd", full(data, version=version, flags=3))


def hdlr(kind):
    return box(b"hdlr", full(u32(0) + kind + bytes(12) + b"Synthetic\0"))


def dinf():
    return box(b"dinf", box(b"dref", full(u32(1) + box(b"url ", full(flags=1)))))


def avc_entry(width, height):
    header = bytes(6) + struct.pack(">H", 1) + bytes(16)
    header += struct.pack(">HHIIIH", width, height, 72 << 16, 72 << 16, 0, 1)
    header += bytes(32) + struct.pack(">Hh", 24, -1)
    # Bounded configuration records, deliberately not decodable SPS/PPS.
    avcc = box(b"avcC", bytes.fromhex("0142001effe10002674201000268ce"))
    return box(b"avc1", header + avcc + box(b"pasp", u32(1, 1)))


def descriptor(tag, data):
    assert len(data) < 128
    # ffmpeg writes four-byte, continuation-encoded descriptor sizes.
    return bytes([tag, 128, 128, 128, len(data)]) + data


def aac_entry():
    asc = descriptor(5, bytes.fromhex("1190"))  # AAC-LC, 48 kHz, stereo
    config = descriptor(4, bytes.fromhex("4015") + bytes(11) + asc)
    es = descriptor(3, b"\0\x02\0" + config + descriptor(6, b"\x02"))
    header = bytes(6) + struct.pack(">H", 1) + bytes(8)
    header += struct.pack(">HHHHI", 2, 16, 0, 0, 48000 << 16)
    return box(b"mp4a", header + box(b"esds", full(es)))


def audio_track(offset):
    # AAC padding: its clock is intentionally longer than the 97-frame video.
    stbl = box(b"stsd", full(u32(1) + aac_entry()))
    stbl += table(b"stts", [(200, 1024)])
    stbl += box(b"stsz", full(u32(4, 200)))
    stbl += table(b"stsc", [(1, 200, 1)]) + table(b"stco", [(offset,)])
    minf = box(b"minf", box(b"smhd", full(bytes(4))) + dinf() + box(b"stbl", stbl))
    mdia = box(b"mdia", clock(b"mdhd", 48000, 204800) + hdlr(b"soun") + minf)
    return box(b"trak", tkhd(2, 4267, 0, 0) + mdia)


@dataclass
class Movie:
    frames: int = 97
    width: int = 768
    height: int = 768
    scale: int = 12288
    delta: int = 512
    chunks: tuple[int, ...] = (48, 49)
    sizes: list[int] | None = None
    fixed_size: bool = False
    co64: bool = False
    wide: bool = False
    version: int = 0
    audio: bool = False
    split_mdat: bool = False
    edits: tuple[tuple[int, int, int], ...] | None = None
    ctts: tuple[tuple[int, int], ...] | None = None
    ctts_version: int = 0
    replace: dict[bytes, bytes] = field(default_factory=dict)
    extra_stbl: bytes = b""
    extra_track: bytes = b""
    extra_moov: bytes = b""
    extra_top: bytes = b""

    def emit(self, kind, payload):
        return self.replace.get(kind, box(kind, payload, wide=self.wide))

    def build(self):
        ftyp = self.emit(b"ftyp", b"isom" + u32(512) + b"isomiso2avc1mp41")
        sizes = self.sizes if self.sizes is not None else [8] * self.frames
        payload = bytes(sum(sizes))
        start = len(ftyp) + (16 if self.wide else 8)
        offsets = []
        consumed = 0
        for count in self.chunks:
            offsets.append(start + sum(sizes[:consumed]))
            consumed += count
        if self.split_mdat:
            assert len(self.chunks) == 2 and not self.audio
            cut = sum(sizes[: self.chunks[0]])
            first = self.emit(b"mdat", payload[:cut])
            gap = box(b"free", b"ignored")
            second = self.emit(b"mdat", payload[cut:])
            offsets[1] += (16 if self.wide else 8) + len(gap)
            mdats = first + gap + second
        else:
            mdats = self.emit(b"mdat", payload + (bytes(800) if self.audio else b""))
        total = self.frames * self.delta
        movie_ticks = (total * 1000 + self.scale - 1) // self.scale
        stbl = self.emit(b"stsd", full(u32(1) + avc_entry(self.width, self.height)))
        stbl += self.replace.get(b"stts", table(b"stts", [(self.frames, self.delta)]))
        size_data = (
            u32(sizes[0], self.frames) if self.fixed_size else u32(0, self.frames) + u32(*sizes)
        )
        stbl += self.emit(b"stsz", full(size_data))
        stbl += self.replace.get(
            b"stsc", table(b"stsc", [(i + 1, n, 1) for i, n in enumerate(self.chunks)])
        )
        offset_kind = b"co64" if self.co64 else b"stco"
        offset_data = b"".join(n.to_bytes(8 if self.co64 else 4, "big") for n in offsets)
        stbl += self.emit(offset_kind, full(u32(len(offsets)) + offset_data))
        if self.ctts is not None:
            stbl += table(
                b"ctts", self.ctts, version=self.ctts_version, signed=self.ctts_version == 1
            )
        stbl += self.extra_stbl
        minf = self.emit(b"vmhd", full(bytes(8), flags=1)) + self.replace.get(b"dinf", dinf())
        minf += self.emit(b"stbl", stbl)
        mdia = self.replace.get(b"mdhd", clock(b"mdhd", self.scale, total, self.version))
        mdia += self.replace.get(b"hdlr", hdlr(b"vide")) + self.emit(b"minf", minf)
        track = self.replace.get(
            b"tkhd", tkhd(1, movie_ticks, self.width, self.height, self.version)
        )
        if self.edits is not None:
            edit_data = u32(len(self.edits))
            for duration, media_time, rate in self.edits:
                size = 4 if self.version == 0 else 8
                edit_data += duration.to_bytes(size, "big") + media_time.to_bytes(
                    size, "big", signed=True
                )
                edit_data += u32(rate)
            track += self.emit(b"edts", self.emit(b"elst", full(edit_data, version=self.version)))
        track += self.emit(b"mdia", mdia) + self.extra_track
        moov = self.replace.get(
            b"mvhd",
            clock(b"mvhd", 1000, max(movie_ticks, 4267 if self.audio else 0), self.version),
        )
        moov += self.emit(b"trak", track)
        if self.audio:
            moov += audio_track(start + len(payload))
        return ftyp + mdats + self.emit(b"moov", moov + self.extra_moov) + self.extra_top


def reject_mp4(data):
    with pytest.raises(media.MeterError, match=r"^invalid or unsupported media$") as error:
        media.mp4_duration(data)
    assert error.value.args == ("invalid or unsupported media",)
    assert error.value.__cause__ is None


def wav(pcm=b"\0\0" * 7, *, channels=1, width=2, rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def encode(data):
    return base64.b64encode(data).decode("ascii")


def reject_wav(value):
    with pytest.raises(media.MeterError, match=r"^invalid or unsupported media$") as error:
        media.wav_frames(value)
    assert error.value.args == ("invalid or unsupported media",)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_wav_actual_frames(channels, width):
    data = wav(bytes(7 * channels * width), channels=channels, width=width)
    frames, rate, got_channels, got_width = media.wav_frames(encode(data))
    assert (frames, rate, got_channels, got_width) == (7, 24000, channels, width)
    assert Fraction(frames, rate) == Fraction(7, 24000)


@pytest.mark.parametrize(
    "value", [None, b"AAAA", "", "not media SECRET", "AAAA\n", "é", "AAAA=", "AB=="]
)
def test_wav_base64_fixed_error(value):
    reject_wav(value)


@pytest.mark.parametrize("pcm,channels", [(b"", 1), (b"\0", 1), (bytes(6), 2)])
def test_wav_empty_and_partial_frames(pcm, channels):
    reject_wav(encode(wav(pcm, channels=channels)))


def test_wav_every_truncation_and_trailing_bytes():
    data = wav()
    for end in range(len(data)):
        reject_wav(encode(data[:end]))
    reject_wav(encode(data + b"secret"))
    # A forged outer length does not hide a truncated inner data chunk.
    truncated = data[:-2]
    truncated = truncated[:4] + struct.pack("<I", len(truncated) - 8) + truncated[8:]
    reject_wav(encode(truncated))


@pytest.mark.parametrize("offset,value", [(20, 3), (22, 0), (24, 0), (28, 0), (32, 1), (34, 12)])
def test_wav_bad_format(offset, value):
    data = bytearray(wav())
    struct.pack_into("<I" if offset in (24, 28) else "<H", data, offset, value)
    reject_wav(encode(data))


def riff(chunks):
    data = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(data)) + data


def wav_chunk(kind, data):
    return kind + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def test_wav_chunks_and_extensible_pcm():
    fmt = wav()[20:36]
    data = wav_chunk(b"data", bytes(14))
    assert (
        media.wav_frames(
            encode(riff([wav_chunk(b"JUNK", b"odd"), wav_chunk(b"fmt ", fmt), data]))
        )[0]
        == 7
    )
    extensible = b"\xfe\xff" + fmt[2:] + struct.pack("<HHI", 22, 16, 4)
    extensible += bytes.fromhex("0100000000001000800000aa00389b71")
    assert media.wav_frames(encode(riff([wav_chunk(b"fmt ", extensible), data])))[0] == 7
    for chunks in (
        [data],
        [data, wav_chunk(b"fmt ", fmt)],
        [wav_chunk(b"fmt ", fmt)] * 2 + [data],
        [wav_chunk(b"fmt ", fmt), data, data],
    ):
        reject_wav(encode(riff(chunks)))
    reject_wav(encode(riff([wav_chunk(b"fmt ", extensible[:-1] + b"\0"), data])))


def test_wav_size_bound():
    data = wav(bytes(media.MAX_WAV_BYTES - 44))
    assert len(data) == media.MAX_WAV_BYTES
    assert media.wav_frames(encode(data))[0] == (media.MAX_WAV_BYTES - 44) // 2
    reject_wav(encode(wav(bytes(media.MAX_WAV_BYTES - 42))))


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize(
    "wide,co64,version", [(False, False, 0), (True, True, 1), (True, False, 0), (False, True, 1)]
)
def test_mp4_exact_video_not_audio_or_movie_duration(audio, wide, co64, version):
    data = Movie(audio=audio, wide=wide, co64=co64, version=version).build()
    assert media.mp4_duration(data) == Fraction(97, 24)
    metadata = media.mp4_metadata(data)
    assert (metadata.frames, metadata.width, metadata.height) == (97, 768, 768)
    assert isinstance(metadata.duration, Fraction)


@pytest.mark.parametrize("fixed_size,split_mdat", [(True, False), (False, True), (True, True)])
def test_mp4_sample_storage_variants(fixed_size, split_mdat):
    movie = Movie(fixed_size=fixed_size, split_mdat=split_mdat)
    if not fixed_size:
        movie.sizes = [i + 1 for i in range(movie.frames)]
    assert media.mp4_duration(movie.build()) == Fraction(97, 24)


def test_mp4_max_samples_and_dimensions():
    movie = Movie(frames=4096, chunks=(4096,), width=3072, height=768)
    assert media.mp4_duration(movie.build()) == Fraction(4096, 24)
    assert media.mp4_metadata(Movie(width=768, height=3072).build()).height == 3072
    reject_mp4(Movie(frames=4097, chunks=(4097,)).build())


@pytest.mark.parametrize("width,height", [(832, 480), (1280, 720)])
@pytest.mark.parametrize("frames", [1, 189])
def test_mp4_cosmos3_super_output_shapes(width, height, frames):
    movie = Movie(frames=frames, chunks=(frames,), width=width, height=height)
    data = movie.build()
    assert media.mp4_duration(data) == Fraction(frames, 24)
    metadata = media.mp4_metadata(data)
    assert (metadata.frames, metadata.width, metadata.height) == (frames, width, height)


@pytest.mark.parametrize(
    "width,height", [(0, 768), (3073, 768), (768, 3073), (3072, 769), (1537, 1536)]
)
def test_mp4_bad_dimensions(width, height):
    reject_mp4(Movie(width=width, height=height).build())


def test_mp4_track_and_sample_entry_dimensions_must_agree():
    reject_mp4(Movie(replace={b"tkhd": tkhd(1, 4042, 640, 480)}).build())


@pytest.mark.parametrize(
    "kind",
    [
        b"ftyp",
        b"moov",
        b"mdat",
        b"mvhd",
        b"trak",
        b"tkhd",
        b"mdia",
        b"hdlr",
        b"mdhd",
        b"minf",
        b"vmhd",
        b"dinf",
        b"stbl",
        b"stsd",
        b"stts",
        b"stsz",
        b"stsc",
        b"stco",
    ],
)
def test_mp4_missing_required_box(kind):
    reject_mp4(Movie(replace={kind: b""}).build())


@pytest.mark.parametrize(
    "kind",
    [
        b"ftyp",
        b"moov",
        b"mvhd",
        b"trak",
        b"tkhd",
        b"mdia",
        b"hdlr",
        b"mdhd",
        b"minf",
        b"vmhd",
        b"dinf",
        b"stbl",
        b"stsd",
        b"stts",
        b"stsz",
        b"stsc",
        b"stco",
    ],
)
def test_mp4_duplicate_box(kind):
    data = Movie().build()
    parser = media._Parser(data)

    def find(boxes):
        for node in boxes:
            if node.kind == kind:
                return data[node.start - 8 : node.end]
            found = find(node.children)
            if found is not None:
                return found
        return None

    original = find(parser.boxes(0, len(data)))
    assert original is not None
    reject_mp4(Movie(replace={kind: original * 2}).build())


def test_mp4_every_truncation():
    data = Movie(frames=5, chunks=(2, 3), audio=True).build()
    for end in range(len(data)):
        reject_mp4(data[:end])
    reject_mp4(data + b"secret")
    # Retain a complete moov, while cutting the mdat short and updating its size.
    parser = media._Parser(data)
    ftyp, mdat, moov = parser.boxes(0, len(data))
    shortened = data[: ftyp.end] + box(b"mdat", bytes(mdat.payload[:39])) + data[moov.start - 8 :]
    reject_mp4(shortened)


@pytest.mark.parametrize(
    "replacement",
    [
        table(b"stts", [(96, 512)]),
        table(b"stts", [(98, 512)]),
        table(b"stts", [(0, 512), (97, 512)]),
        table(b"stts", [(97, 0)]),
        table(b"stts", [(97, 513)]),
        table(b"stts", [(0xFFFFFFFF, 512)]),
        box(b"stts", full(u32(0xFFFFFFFF))),
        box(b"stts", full(u32(4097))),
        box(b"stts", full(u32(0))),
        box(b"stts", full(u32(1, 97, 512, 7))),
        box(b"stts", full(u32(1, 97, 512), version=1)),
    ],
)
def test_mp4_bad_stts(replacement):
    reject_mp4(Movie(replace={b"stts": replacement}).build())


@pytest.mark.parametrize(
    "replacement",
    [
        box(b"stsz", full(u32(0, 96) + u32(*([8] * 96)))),
        box(b"stsz", full(u32(0, 97) + u32(*([8] * 96)))),
        box(b"stsz", full(u32(0, 97) + u32(*([8] * 96 + [0])))),
        box(b"stsz", full(u32(0, 0))),
        box(b"stsz", full(u32(8, 0xFFFFFFFF))),
        box(b"stsz", full(u32(8, 4097))),
        box(b"stsz", full(u32(0xFFFFFFFF, 97))),
        box(b"stsz", full(u32(8, 97, 0))),
    ],
)
def test_mp4_bad_stsz(replacement):
    reject_mp4(Movie(replace={b"stsz": replacement}).build())


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(0, 97, 1)],
        [(2, 97, 1)],
        [(1, 0, 1)],
        [(1, 97, 0)],
        [(1, 97, 2)],
        [(1, 48, 1)],
        [(1, 49, 1)],
        [(1, 48, 1), (1, 49, 1)],
        [(1, 48, 1), (3, 49, 1)],
        [(1, 0xFFFFFFFF, 1)],
    ],
)
def test_mp4_bad_chunk_layout(rows):
    reject_mp4(Movie(replace={b"stsc": table(b"stsc", rows)}).build())


@pytest.mark.parametrize(
    "offsets",
    [[0, 424], [32, 424], [40, 40], [40, 420], [40, 816], [40, 0xFFFFFFFF], [40], [40, 424, 432]],
)
def test_mp4_outside_mdat_or_overlapping_chunks(offsets):
    # Normal ftyp=32 bytes, mdat payload starts at 40; second chunk starts at 424.
    reject_mp4(Movie(replace={b"stco": table(b"stco", [(n,) for n in offsets])}).build())


def test_mp4_chunks_can_be_physically_reordered_without_overlap():
    movie = Movie(chunks=(49, 48), replace={b"stco": table(b"stco", [(424,), (40,)])})
    assert media.mp4_duration(movie.build()) == Fraction(97, 24)


def test_mp4_64bit_out_of_range_and_dual_offset_tables():
    huge = box(b"co64", full(u32(2) + struct.pack(">QQ", 40, (1 << 64) - 1)))
    reject_mp4(Movie(co64=True, replace={b"co64": huge}).build())
    reject_mp4(Movie(extra_stbl=box(b"co64", full(u32(2) + struct.pack(">QQ", 40, 424)))).build())
    reject_mp4(Movie(replace={b"stco": box(b"stco", full(u32(0xFFFFFFFF)))}).build())


@pytest.mark.parametrize(
    "kind,scale,duration",
    [
        (b"mdhd", 0, 49664),
        (b"mdhd", 12288, 0),
        (b"mdhd", 12288, 49665),
        (b"mdhd", 12288, 0xFFFFFFFF),
        (b"mvhd", 0, 4042),
        (b"mvhd", 1000, 4000),
    ],
)
def test_mp4_inconsistent_clocks(kind, scale, duration):
    reject_mp4(Movie(replace={kind: clock(kind, scale, duration)}).build())


def test_mp4_millisecond_rounding_is_not_the_meter():
    assert media.mp4_duration(Movie().build()) != Fraction(4042, 1000)
    reject_mp4(Movie(replace={b"tkhd": tkhd(1, 4041, 768, 768)}).build())
    # Reject 30 fps and 23.976 fps rather than guessing the requested frame rate.
    reject_mp4(Movie(scale=15360).build())
    reject_mp4(Movie(scale=24000, delta=1001).build())


@pytest.mark.parametrize("version", [0, 1])
def test_mp4_single_full_length_b_frame_edit(version):
    # Presentation sample order: 0, 3, 1, 2, 4; all five frames remain present.
    ctts = ((1, 1024), (1, 2048), (2, 512), (1, 1024))
    movie = Movie(frames=5, chunks=(2, 3), ctts=ctts, edits=((209, 1024, 65536),), version=version)
    assert media.mp4_duration(movie.build()) == Fraction(5, 24)
    # A zero-offset, full-length edit is also ordinary ffmpeg output.
    assert media.mp4_duration(
        Movie(edits=((4042, 0, 65536),), version=version).build()
    ) == Fraction(97, 24)


def test_mp4_signed_composition_offsets():
    ctts = ((1, 0), (1, 1024), (2, -512), (1, 0))
    movie = Movie(frames=5, chunks=(2, 3), ctts=ctts, ctts_version=1)
    assert media.mp4_duration(movie.build()) == Fraction(5, 24)


@pytest.mark.parametrize(
    "edits",
    [
        (),
        ((4041, 0, 65536),),
        ((4043, 0, 65536),),
        ((4042, -1, 65536),),
        ((4042, 512, 65536),),
        ((4042, 0, 0),),
        ((4042, 0, 65537),),
        ((4042, 0, 131072),),
        ((1000, -1, 65536), (4042, 0, 65536)),
    ],
)
def test_mp4_rejects_trimming_gaps_and_complex_edits(edits):
    reject_mp4(Movie(edits=edits).build())


@pytest.mark.parametrize(
    "ctts,edits",
    [
        (((96, 0),), None),
        (((98, 0),), None),
        (((0, 0), (97, 0)), None),
        (((97, 512),), None),
        (((97, 0xFFFFFFFF),), None),
        (((1, 512), (96, 0)), ((4042, 0, 65536),)),
        (((1, 0), (96, 512)), None),
    ],
)
def test_mp4_composition_count_overlap_or_gap(ctts, edits):
    reject_mp4(Movie(ctts=ctts, edits=edits).build())


@pytest.mark.parametrize("fragment", [b"moof", b"mvex", b"mfra", b"styp", b"sidx", b"traf"])
def test_mp4_fragmented_rejected(fragment):
    reject_mp4(Movie(extra_top=box(fragment)).build())
    reject_mp4(Movie(extra_moov=box(fragment)).build())


@pytest.mark.parametrize(
    "extra",
    [
        table(b"stss", [(1,), (98,)]),
        table(b"stss", [(1,), (1,)]),
        table(b"stss", [(0,)]),
        box(b"sdtp", full(bytes(96))),
        box(b"stz2"),
        box(b"senc"),
        box(b"unknown"),
    ],
)
def test_mp4_unsupported_or_inconsistent_auxiliary_tables(extra):
    reject_mp4(Movie(extra_stbl=extra).build())


def test_mp4_supported_auxiliary_tables():
    extra = table(b"stss", [(1,), (49,)]) + box(b"sdtp", full(bytes(97)))
    assert media.mp4_duration(Movie(extra_stbl=extra).build()) == Fraction(97, 24)


def test_mp4_unsupported_codec_external_reference_and_nonvideo():
    data = Movie().build()
    reject_mp4(data.replace(b"avc1", b"hvc1"))
    reject_mp4(data.replace(b"avcC", b"hvcC"))
    reject_mp4(data.replace(bytes.fromhex("0142001effe1"), bytes.fromhex("0142001efee1")))
    external = box(b"dinf", box(b"dref", full(u32(1) + box(b"url ", full(b"secret\0")))))
    reject_mp4(Movie(replace={b"dinf": external}).build())
    reject_mp4(Movie(replace={b"hdlr": hdlr(b"hint")}).build())
    reject_mp4(Movie(audio=True).build().replace(b"vide", b"soun"))
    reject_mp4(Movie(audio=True).build().replace(b"mp4a", b"ac-3"))
    reject_mp4(Movie(audio=True).build().replace(bytes.fromhex("4015"), bytes.fromhex("2015")))


@pytest.mark.parametrize(
    "raw",
    [
        u32(2) + b"free",
        u32(7) + b"free",
        u32(0xFFFFFFFF) + b"free",
        u32(1) + b"free",
        u32(1) + b"free" + struct.pack(">Q", 8),
        u32(1) + b"free" + struct.pack(">Q", (1 << 64) - 1),
    ],
)
def test_mp4_pathological_box_sizes(raw):
    reject_mp4(Movie(extra_top=raw).build())


def test_mp4_size_zero_and_resource_limits(monkeypatch):
    # Only top-level boxes may use EOF size zero.
    data = Movie().build()
    assert media.mp4_duration(data + u32(0) + b"free" + b"opaque") == Fraction(97, 24)
    reject_mp4(Movie(extra_stbl=u32(0) + b"stts").build())
    nested = box(b"free")
    for _ in range(14):
        nested = box(b"udta", nested)
    reject_mp4(Movie(extra_moov=nested).build())
    monkeypatch.setattr(media, "_MAX_BOXES", 20)
    reject_mp4(Movie(extra_top=box(b"free") * 21).build())
    monkeypatch.setattr(media, "MAX_MP4_BYTES", len(data) - 1)
    reject_mp4(data)


def test_mp4_bad_input_types():
    for value in (None, "secret", bytearray(b"secret"), memoryview(b"secret"), b""):
        reject_mp4(value)
