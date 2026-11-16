"""Metadata extraction using FFprobe for video inspection."""

import json
import logging
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scripts.utils import (
    CorruptedVideoError,
    InvalidVideoError,
    format_bytes,
    seconds_to_hms,
    verify_ffmpeg,
)


def _parse_rational(val_str: Optional[str]) -> Optional[float]:
    """Parse rational string like '30/1' or '30000/1001' into float."""
    if not val_str:
        return None
    try:
        if "/" in val_str:
            num, den = val_str.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        return float(val_str)
    except (ValueError, ZeroDivisionError):
        return None


def _calculate_aspect_ratio(width: int, height: int) -> str:
    """Calculate simplified aspect ratio string e.g. 16:9 or 4:3."""
    if width <= 0 or height <= 0:
        return "unknown"
    gcd = math.gcd(width, height)
    w_ratio = width // gcd
    h_ratio = height // gcd
    # If ratio numbers are too large (e.g. odd dimensions), provide approximate standard
    if w_ratio > 30 or h_ratio > 30:
        ratio = width / height
        if abs(ratio - 16 / 9) < 0.05:
            return "16:9"
        if abs(ratio - 4 / 3) < 0.05:
            return "4:3"
        if abs(ratio - 21 / 9) < 0.05:
            return "21:9"
        if abs(ratio - 1.0) < 0.05:
            return "1:1"
        return f"{width}:{height}"
    return f"{w_ratio}:{h_ratio}"


class MetadataExtractor:
    """Inspects video files and extracts detailed metadata via FFprobe."""

    def __init__(self, ffprobe_path: Optional[str] = None):
        if ffprobe_path:
            self.ffprobe_path = ffprobe_path
        else:
            _, self.ffprobe_path = verify_ffmpeg()

    def inspect(self, video_path: Path | str) -> Dict[str, Any]:
        """Inspect a video file and return complete structured metadata.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary containing extracted metadata fields.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            CorruptedVideoError: If FFprobe encounters an error parsing the file.
            InvalidVideoError: If the file has no video stream.
        """
        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise InvalidVideoError(f"Video file is empty (0 bytes): {path.name}")

        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_error",
            str(path),
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            probe_data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise CorruptedVideoError(f"FFprobe failed to inspect {path.name}: {e.stderr}") from e
        except json.JSONDecodeError as e:
            raise CorruptedVideoError(f"FFprobe returned invalid JSON for {path.name}") from e

        if "error" in probe_data:
            err_msg = probe_data["error"].get("string", "Unknown FFprobe error")
            raise CorruptedVideoError(f"FFprobe error on {path.name}: {err_msg}")

        format_info = probe_data.get("format", {})
        streams = probe_data.get("streams", [])

        # Filter video streams
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            raise InvalidVideoError(f"File contains no decodable video stream: {path.name}")

        primary_video = video_streams[0]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        # Parse duration
        duration_sec: Optional[float] = None
        if "duration" in format_info:
            try:
                duration_sec = float(format_info["duration"])
            except (ValueError, TypeError):
                pass
        if duration_sec is None and "duration" in primary_video:
            try:
                duration_sec = float(primary_video["duration"])
            except (ValueError, TypeError):
                pass

        # Width and height
        width = primary_video.get("width")
        height = primary_video.get("height")

        # Display Aspect Ratio
        dar = primary_video.get("display_aspect_ratio")
        if not dar and width and height:
            dar = _calculate_aspect_ratio(width, height)

        # Frame rate
        fps = _parse_rational(primary_video.get("avg_frame_rate"))
        if not fps or fps == 0.0:
            fps = _parse_rational(primary_video.get("r_frame_rate"))

        # Frame count
        total_frame_count: Optional[int] = None
        nb_frames_str = primary_video.get("nb_frames")
        if nb_frames_str and nb_frames_str.isdigit():
            total_frame_count = int(nb_frames_str)
        elif duration_sec is not None and fps is not None and fps > 0:
            # Note: Estimated frame count when container does not store nb_frames header
            total_frame_count = int(round(duration_sec * fps))

        # Audio stream info
        primary_audio = audio_streams[0] if audio_streams else {}
        audio_codec = primary_audio.get("codec_name")
        audio_sample_rate = None
        if primary_audio.get("sample_rate"):
            try:
                audio_sample_rate = int(primary_audio["sample_rate"])
            except (ValueError, TypeError):
                pass
        audio_channels = primary_audio.get("channels")

        # Creation timestamp
        tags = format_info.get("tags", {})
        creation_timestamp = tags.get("creation_time") or primary_video.get("tags", {}).get("creation_time")

        metadata: Dict[str, Any] = {
            "original_filename": path.name,
            "file_path": str(path),
            "file_size_bytes": file_size,
            "file_size_human": format_bytes(file_size),
            "container_format": format_info.get("format_name"),
            "duration_seconds": duration_sec,
            "duration_formatted": seconds_to_hms(duration_sec, include_ms=True) if duration_sec is not None else None,
            "width": width,
            "height": height,
            "display_aspect_ratio": dar,
            "frame_rate": round(fps, 4) if fps else None,
            "total_frame_count": total_frame_count,
            "video_codec": primary_video.get("codec_name"),
            "audio_codec": audio_codec,
            "audio_sample_rate": audio_sample_rate,
            "audio_channels": audio_channels,
            "audio_stream_count": len(audio_streams),
            "audio_streams": [
                {
                    "index": s.get("index"),
                    "codec": s.get("codec_name"),
                    "sample_rate": s.get("sample_rate"),
                    "channels": s.get("channels"),
                    "language": s.get("tags", {}).get("language"),
                }
                for s in audio_streams
            ],
            "creation_timestamp": creation_timestamp,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_extraction_interval": None,
            "total_extracted_images": 0,
            "processing_duration_seconds": 0.0,
            "processing_status": "PENDING",
            "error_message": None,
        }

        return metadata

    @staticmethod
    def save(metadata: Dict[str, Any], output_path: Path | str) -> None:
        """Save metadata dictionary as JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

