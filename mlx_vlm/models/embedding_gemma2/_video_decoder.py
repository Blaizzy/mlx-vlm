"""Optional planar video decoding through OpenCV's bundled FFmpeg libraries.

Only the FFmpeg 7 public ABI below is supported. Library discovery and version
checks run lazily, before allocating or dereferencing native structures. Other
platforms, builds and video layouts use the existing OpenCV decoder instead.
The struct prefixes follow FFmpeg n7.1.1's public headers in libavformat,
libavcodec and libavutil; never allocate these structures from Python.
"""

import ctypes as C
import errno
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np


class VideoDecodeUnavailable(RuntimeError):
    """The optional decoder cannot handle this environment or video."""


P = C.c_void_p
I = C.c_int
L = C.c_int64
B = C.POINTER(C.c_uint8)


class Rational(C.Structure):
    _fields_ = [("num", I), ("den", I)]


class Packet(C.Structure):
    _fields_ = [
        ("buf", P),
        ("pts", L),
        ("dts", L),
        ("data", B),
        ("size", I),
        ("stream_index", I),
        ("flags", I),
        ("side_data", P),
        ("side_data_elems", I),
        ("duration", L),
        ("pos", L),
        ("opaque", P),
        ("opaque_ref", P),
        ("time_base", Rational),
    ]


class Stream(C.Structure):
    _fields_ = [
        ("av_class", P),
        ("index", I),
        ("id", I),
        ("codecpar", P),
        ("priv_data", P),
        ("time_base", Rational),
        ("start_time", L),
        ("duration", L),
        ("nb_frames", L),
    ]


class Format(C.Structure):
    _fields_ = [
        ("av_class", P),
        ("iformat", P),
        ("oformat", P),
        ("priv_data", P),
        ("pb", P),
        ("ctx_flags", I),
        ("nb_streams", C.c_uint),
        ("streams", C.POINTER(C.POINTER(Stream))),
    ]


class Frame(C.Structure):
    _fields_ = [
        ("data", B * 8),
        ("linesize", I * 8),
        ("extended_data", C.POINTER(B)),
        ("width", I),
        ("height", I),
        ("nb_samples", I),
        ("format", I),
        ("key_frame", I),
        ("pict_type", I),
        ("sample_aspect_ratio", Rational),
        ("pts", L),
        ("pkt_dts", L),
        ("time_base", Rational),
        ("quality", I),
        ("opaque", P),
        ("repeat_pict", I),
        ("interlaced_frame", I),
        ("top_field_first", I),
        ("palette_has_changed", I),
        ("sample_rate", I),
        ("buf", P * 8),
        ("extended_buf", P),
        ("nb_extended_buf", I),
        ("side_data", P),
        ("nb_side_data", I),
        ("flags", I),
        ("color_range", I),
        ("color_primaries", I),
        ("color_trc", I),
        ("colorspace", I),
        ("chroma_location", I),
        ("best_effort_timestamp", L),
    ]


class Component(C.Structure):
    _fields_ = [("plane", I), ("step", I), ("offset", I), ("shift", I), ("depth", I)]


class PixelDescriptor(C.Structure):
    _fields_ = [
        ("name", C.c_char_p),
        ("nb_components", C.c_uint8),
        ("log2_chroma_w", C.c_uint8),
        ("log2_chroma_h", C.c_uint8),
        ("flags", C.c_uint64),
        ("comp", Component * 4),
        ("alias", C.c_char_p),
    ]


class CodecParameters(C.Structure):
    _fields_ = [
        ("codec_type", I),
        ("codec_id", I),
        ("codec_tag", C.c_uint),
        ("extradata", B),
        ("extradata_size", I),
        ("coded_side_data", P),
        ("nb_coded_side_data", I),
    ]


class InputFormat(C.Structure):
    _fields_ = [("name", C.c_char_p)]


class _FFmpeg:
    def __init__(self, root):
        libraries = []
        for stem, major in (("avformat", 61), ("avcodec", 61), ("avutil", 59)):
            paths = {p.resolve() for p in root.glob(f"lib{stem}*.dylib")}
            if len(paths) != 1:
                raise VideoDecodeUnavailable(f"Missing or ambiguous bundled {stem}")
            lib = C.CDLL(str(paths.pop()))
            version = getattr(lib, stem + "_version")
            version.argtypes, version.restype = [], C.c_uint
            if version() >> 16 != major:
                raise VideoDecodeUnavailable(
                    f"Unsupported {stem} ABI (expected {major})"
                )
            libraries.append(lib)
        self.libraries = libraries  # Keep every library alive with its functions.
        fmt, codec, util = libraries
        self.bind(fmt, "avformat_open_input", I, C.POINTER(P), C.c_char_p, P, P)
        self.bind(fmt, "avformat_find_stream_info", I, P, P)
        self.bind(fmt, "av_find_best_stream", I, P, I, I, I, C.POINTER(P), I)
        self.bind(fmt, "av_read_frame", I, P, P)
        self.bind(fmt, "av_seek_frame", I, P, I, L, I)
        self.bind(fmt, "avformat_close_input", None, C.POINTER(P))
        self.bind(codec, "av_packet_alloc", P)
        self.bind(codec, "av_packet_unref", None, P)
        self.bind(codec, "av_packet_free", None, C.POINTER(P))
        self.bind(codec, "avcodec_alloc_context3", P, P)
        self.bind(codec, "avcodec_parameters_to_context", I, P, P)
        self.bind(codec, "avcodec_open2", I, P, P, C.POINTER(P))
        self.bind(codec, "avcodec_send_packet", I, P, P)
        self.bind(codec, "avcodec_receive_frame", I, P, P)
        self.bind(codec, "avcodec_flush_buffers", None, P)
        self.bind(codec, "avcodec_free_context", None, C.POINTER(P))
        self.bind(util, "av_frame_alloc", P)
        self.bind(util, "av_frame_unref", None, P)
        self.bind(util, "av_frame_free", None, C.POINTER(P))
        self.bind(util, "av_get_pix_fmt_name", C.c_char_p, I)
        self.bind(util, "av_get_pix_fmt", I, C.c_char_p)
        self.bind(util, "av_pix_fmt_desc_get", C.POINTER(PixelDescriptor), I)
        self.bind(util, "av_dict_set", I, C.POINTER(P), C.c_char_p, C.c_char_p, I)
        self.bind(util, "av_dict_free", None, C.POINTER(P))
        self.bind(util, "av_strerror", I, I, C.c_char_p, C.c_size_t)
        self.bind(util, "av_frame_get_side_data", P, P, I)
        self.bind(codec, "av_packet_side_data_get", P, P, I, I)

    def bind(self, library, name, restype, *argtypes):
        fn = getattr(library, name)
        fn.restype, fn.argtypes = restype, argtypes
        setattr(self, name, fn)

    def check(self, result):
        if result < 0:
            buffer = C.create_string_buffer(256)
            self.av_strerror(result, buffer, len(buffer))
            raise VideoDecodeUnavailable(
                f"FFmpeg error {result}: {buffer.value.decode(errors='replace')}"
            )
        return result


@lru_cache(maxsize=1)
def _load_ffmpeg():
    import mlx.core as mx

    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or not mx.metal.is_available()
    ):
        raise VideoDecodeUnavailable("Metal video decoding requires Apple Silicon")
    import cv2

    package = Path(cv2.__file__).resolve().parent
    # Discover a complete bundle together; never mix unrelated system libraries.
    for root in (package / ".dylibs", package.parent / "opencv_python.libs"):
        if root.is_dir():
            try:
                return _FFmpeg(root)
            except (OSError, AttributeError) as exc:
                raise VideoDecodeUnavailable(
                    "Cannot load bundled FFmpeg symbols"
                ) from exc
    raise VideoDecodeUnavailable("OpenCV does not expose bundled FFmpeg libraries")


EOF = -541478725
AGAIN = -errno.EAGAIN
NO_PTS = -(1 << 63)


class Decoder:
    def __init__(self, path):
        self.path = str(path)
        self.context = P()
        self.codec_context = P()
        self.packet = P()
        self.frame = P()
        self.api = _load_ffmpeg()
        try:
            self.api.check(
                self.api.avformat_open_input(
                    C.byref(self.context), self.path.encode(), None, None
                )
            )
            self.api.check(self.api.avformat_find_stream_info(self.context, None))
            decoder = P()
            self.index = self.api.check(
                self.api.av_find_best_stream(
                    self.context, 0, -1, -1, C.byref(decoder), 0
                )
            )
            container = C.cast(self.context, C.POINTER(Format)).contents
            stream = container.streams[self.index].contents
            params = C.cast(stream.codecpar, C.POINTER(CodecParameters)).contents
            if self.api.av_packet_side_data_get(
                params.coded_side_data, params.nb_coded_side_data, 5
            ):
                raise VideoDecodeUnavailable("Video has a display transform")
            self.is_mov = b"mov" in C.cast(
                container.iformat, C.POINTER(InputFormat)
            ).contents.name.split(b",")
            self.time_base = (stream.time_base.num, stream.time_base.den)
            if min(self.time_base) <= 0:
                raise VideoDecodeUnavailable("Video has no valid time base")
            self.codec_context = P(self.api.avcodec_alloc_context3(decoder))
            if not self.codec_context:
                raise VideoDecodeUnavailable("Cannot allocate video codec")
            self.api.check(
                self.api.avcodec_parameters_to_context(
                    self.codec_context, stream.codecpar
                )
            )
            options = P()
            self.api.av_dict_set(C.byref(options), b"threads", b"4", 0)
            try:
                self.api.check(
                    self.api.avcodec_open2(
                        self.codec_context, decoder, C.byref(options)
                    )
                )
            finally:
                self.api.av_dict_free(C.byref(options))
            self.packet = P(self.api.av_packet_alloc())
            self.frame = P(self.api.av_frame_alloc())
            if not self.packet or not self.frame:
                raise VideoDecodeUnavailable("Cannot allocate video buffers")
            self.timeline = self.scan()
        except BaseException:
            self.close()
            raise

    def scan(self):
        packets = []
        while True:
            status = self.api.av_read_frame(self.context, self.packet)
            if status == EOF:
                break
            self.api.check(status)
            packet = C.cast(self.packet, C.POINTER(Packet)).contents
            if packet.stream_index == self.index and not packet.flags & 4:
                pts = packet.pts if packet.pts != NO_PTS else packet.dts
                packets.append((int(pts), int(packet.duration), int(packet.dts)))
            self.api.av_packet_unref(self.packet)
        if not packets or any(p[0] == NO_PTS or p[1] <= 0 for p in packets):
            raise VideoDecodeUnavailable("Video has incomplete packet timestamps")
        if len({p[0] for p in packets}) != len(packets):
            raise VideoDecodeUnavailable("Video has ambiguous packet timestamps")
        # The bundled FFmpeg 7 MOV demuxer flattens packet durations for some
        # VFR files. Recover their decode-time intervals when DTS demonstrates
        # variable timing but all reported packet durations are constant. This
        # uses only file timestamps, not reference frame selections.
        deltas = [b[2] - a[2] for a, b in zip(packets, packets[1:])]
        self.recovered_packet_durations = False
        if (
            self.is_mov
            and len({p[1] for p in packets}) == 1
            and len(set(deltas)) > 1
            and all(p[2] != NO_PTS for p in packets)
            and all(d > 0 for d in deltas)
        ):
            packets = [
                (p[0], deltas[i] if i < len(deltas) else p[1], p[2])
                for i, p in enumerate(packets)
            ]
            self.recovered_packet_durations = True
        packets = [(p[0], p[1]) for p in packets]
        packets.sort()
        begin = min(p[0] for p in packets)
        end = max(p[0] + p[1] for p in packets)
        # Match FFmpeg's av_q2d followed by timestamp multiplication.
        tb = self.time_base[0] / self.time_base[1]
        duration = end * tb - begin * tb
        self.metadata = dict(
            total_num_frames=len(packets),
            fps=len(packets) / duration,
            duration=duration,
        )
        return packets

    def get(self, index):
        target = self.timeline[index][0]
        self.api.check(self.api.av_seek_frame(self.context, self.index, target, 1))
        self.api.avcodec_flush_buffers(self.codec_context)
        self.api.av_packet_unref(self.packet)
        self.api.av_frame_unref(self.frame)
        flushing = False
        while True:
            if not flushing:
                status = self.api.av_read_frame(self.context, self.packet)
                if status == EOF:
                    flushing = True
                    self.api.check(
                        self.api.avcodec_send_packet(self.codec_context, None)
                    )
                else:
                    self.api.check(status)
                    packet = C.cast(self.packet, C.POINTER(Packet)).contents
                    if packet.stream_index != self.index:
                        self.api.av_packet_unref(self.packet)
                        continue
                    self.api.check(
                        self.api.avcodec_send_packet(self.codec_context, self.packet)
                    )
                    self.api.av_packet_unref(self.packet)
            while True:
                status = self.api.avcodec_receive_frame(self.codec_context, self.frame)
                if status == AGAIN:
                    break
                if status == EOF:
                    raise VideoDecodeUnavailable(
                        f"Frame {index}, PTS {target} not found"
                    )
                self.api.check(status)
                frame = C.cast(self.frame, C.POINTER(Frame)).contents
                pts = frame.pts if frame.pts != NO_PTS else frame.best_effort_timestamp
                if pts >= target:
                    if pts != target:
                        raise VideoDecodeUnavailable(
                            f"Requested PTS {target}, decoded {pts}"
                        )
                    return self.copy_frame(frame)
                self.api.av_frame_unref(self.frame)
            if flushing:
                raise VideoDecodeUnavailable("Decoder stopped before requested frame")

    def copy_frame(self, frame):
        if frame.width <= 0 or frame.height <= 0:
            raise VideoDecodeUnavailable("Invalid frame dimensions")
        if frame.width % 2 or frame.height % 2:
            raise VideoDecodeUnavailable("Odd frame dimensions use OpenCV")
        if frame.flags & 8 or frame.interlaced_frame or frame.color_trc in (16, 18):
            raise VideoDecodeUnavailable("Interlaced and HDR frames use OpenCV")
        if self.api.av_frame_get_side_data(self.frame, 6):
            raise VideoDecodeUnavailable("Frame has a display transform")
        descriptor = self.api.av_pix_fmt_desc_get(frame.format)
        if not descriptor:
            raise VideoDecodeUnavailable("Unknown pixel format")
        desc = descriptor.contents
        name = desc.name.decode()
        if name not in (
            "yuv420p",
            "yuvj420p",
            "yuv422p",
            "yuvj422p",
            "yuv420p10le",
            "yuv422p10le",
        ):
            raise VideoDecodeUnavailable(f"Unsupported pixel format: {name}")
        depth = desc.comp[0].depth
        planes = []
        for component in range(3):
            c = desc.comp[component]
            sx = desc.log2_chroma_w if component else 0
            sy = desc.log2_chroma_h if component else 0
            w = (frame.width + (1 << sx) - 1) >> sx
            h = (frame.height + (1 << sy) - 1) >> sy
            stride = frame.linesize[c.plane]
            if stride < w * c.step or not frame.data[c.plane]:
                raise VideoDecodeUnavailable("Unsupported plane stride")
            raw = np.ctypeslib.as_array(frame.data[c.plane], shape=(h * stride,))
            dtype = np.dtype("uint8" if depth == 8 else "<u2")
            arr = np.ndarray(
                (h, w),
                dtype=dtype,
                buffer=raw,
                strides=(stride, c.step),
                offset=c.offset,
            )
            planes.append(arr.copy())
        meta = dict(
            width=frame.width,
            height=frame.height,
            format=name,
            format_id=frame.format,
            depth=depth,
            log2_chroma_w=desc.log2_chroma_w,
            log2_chroma_h=desc.log2_chroma_h,
            colorspace=frame.colorspace,
            color_range=frame.color_range,
            chroma_location=frame.chroma_location,
            pts=frame.pts,
        )
        return planes, meta

    def close(self):
        if self.frame:
            self.api.av_frame_free(C.byref(self.frame))
        if self.packet:
            self.api.av_packet_free(C.byref(self.packet))
        if self.codec_context:
            self.api.avcodec_free_context(C.byref(self.codec_context))
        if self.context:
            self.api.avformat_close_input(C.byref(self.context))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def decode_video(path, sampler):
    """Decode selected frames as HWC RGB, or request the OpenCV fallback."""
    from transformers.video_utils import VideoMetadata

    from ._video_kernel import yuv_to_rgb

    path = str(path)
    if path.startswith("file://"):
        path = path[7:]
    if not Path(path).is_file():
        raise VideoDecodeUnavailable("Native decoding requires a local video file")
    with Decoder(path) as decoder:
        metadata = VideoMetadata(**decoder.metadata)
        indices = np.asarray(
            (
                sampler(metadata=metadata)
                if sampler is not None
                else np.arange(metadata.total_num_frames)
            ),
            dtype=int,
        ).reshape(-1)
        if (
            not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= metadata.total_num_frames)
        ):
            raise ValueError("Frame indices must be within a non-empty video")
        frames = []
        for index in indices:
            planes, info = decoder.get(int(index))
            try:
                frame = np.asarray(yuv_to_rgb(planes, info))
            except RuntimeError as exc:
                raise VideoDecodeUnavailable(
                    "Metal color conversion unavailable"
                ) from exc
            if frames and frame.shape != frames[0].shape:
                raise VideoDecodeUnavailable("Video changes frame dimensions")
            frames.append(frame)
        video = np.stack(frames)
        metadata.frames_indices = indices
        metadata.width, metadata.height = video.shape[2], video.shape[1]
        metadata.video_backend = "ffmpeg_metal"
        return video, metadata
