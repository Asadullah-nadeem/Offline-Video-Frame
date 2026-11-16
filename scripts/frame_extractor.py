"""Frame extraction engine supporting Interval, All-Frames, Custom Timestamps, and Scene Changes."""

import csv
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2

from scripts.file_manager import check_disk_space
from scripts.utils import (
    DiskSpaceError,
    VideoProcessingError,
    format_bytes,
    parse_timestamp,
    seconds_to_hms,
    verify_ffmpeg,
)

logger = logging.getLogger("VideoProcessor.FrameExtractor")


def build_scale_filter(
    target_dim: Optional[str],
    preserve_resolution: bool = True,
) -> Optional[str]:
    """Build FFmpeg scale filter string respecting aspect ratio and preventing upscaling.
    
    Args:
        target_dim: Dimensions string e.g. '1920x1080', '1280x720', or None.
        preserve_resolution: If True and target_dim is None, no scaling is performed.
        
    Returns:
        Filter string for -vf, or None if no scaling is needed.
    """
    if preserve_resolution and not target_dim:
        return None

    if not target_dim:
        return None

    # Parse WxH
    m = re.match(r"^(\d+)\s*[xX:]\s*(\d+)$", target_dim.strip())
    if not m:
        logger.warning(f"Invalid resize format '{target_dim}'. Expected 'WIDTHxHEIGHT' (e.g. '1920x1080'). Ignoring.")
        return None

    target_w = int(m.group(1))
    target_h = int(m.group(2))

    # Scale to fit within bounding box, preserve aspect ratio, never upscale, ensure even dimensions
    # scale='min(iw,TARGET_W)':'min(ih,TARGET_H)':force_original_aspect_ratio=decrease
    return (
        f"scale='trunc(min(iw,{target_w})/2)*2':'trunc(min(ih,{target_h})/2)*2':"
        f"force_original_aspect_ratio=decrease:flags=lanczos"
    )


def get_codec_and_quality_args(
    image_format: str,
    jpeg_quality: int = 95,
) -> List[str]:
    """Generate FFmpeg encoding arguments based on format and quality settings."""
    fmt = image_format.lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        # FFmpeg mjpeg quality range: 2 (best ~95-100%) to 31 (worst)
        # Map 1-100 quality where 100 -> 2, 1 -> 31
        q_val = max(2, min(31, round(31 - (jpeg_quality - 1) * 29 / 99)))
        return ["-c:v", "mjpeg", "-q:v", str(q_val), "-pix_fmt", "yuvj420p", "-strict", "-2"]
    elif fmt == "png":
        return ["-c:v", "png", "-compression_level", "6"]
    elif fmt == "webp":
        q_val = max(1, min(100, jpeg_quality))
        return ["-c:v", "libwebp", "-quality", str(q_val)]
    else:
        # Fallback to extension default
        return []


def validate_image_cv2(image_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """Validate that an extracted image file exists, is non-empty, and can be decoded by OpenCV.
    
    Returns:
        (is_valid, error_message, (width, height))
    """
    if not image_path.is_file():
        return False, f"Image file not found: {image_path.name}", None

    if image_path.stat().st_size == 0:
        return False, f"Image file is empty (0 bytes): {image_path.name}", None

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, f"OpenCV failed to decode image: {image_path.name}", None

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return False, f"Decoded image has invalid dimensions ({w}x{h}): {image_path.name}", None

        return True, None, (w, h)
    except Exception as e:
        return False, f"Validation exception: {e}", None


class FrameExtractor:
    """Orchestrates video frame extraction across various modes with image validation and CSV mapping."""

    def __init__(self, ffmpeg_path: Optional[str] = None, ffprobe_path: Optional[str] = None):
        if ffmpeg_path and ffprobe_path:
            self.ffmpeg_path = ffmpeg_path
            self.ffprobe_path = ffprobe_path
        else:
            self.ffmpeg_path, self.ffprobe_path = verify_ffmpeg()

    def _write_csv(self, records: List[Dict[str, Any]], csv_path: Path) -> None:
        """Write frame information to CSV file."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["frame_number", "timestamp_seconds", "timestamp", "filename"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "frame_number": r["frame_number"],
                    "timestamp_seconds": r["timestamp_seconds"],
                    "timestamp": r["timestamp"],
                    "filename": r["filename"],
                })

    def extract_interval(
        self,
        video_path: Path | str,
        frames_dir: Path | str,
        csv_path: Path | str,
        metadata: Dict[str, Any],
        interval_seconds: float = 60.0,
        include_end_frame: bool = False,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        resize: Optional[str] = None,
        preserve_resolution: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract frames at fixed time intervals (Mode A).
        
        Args:
            video_path: Input video file.
            frames_dir: Directory to save extracted images.
            csv_path: Path to save frames.csv.
            metadata: Video metadata dictionary.
            interval_seconds: Time interval in seconds between extractions.
            include_end_frame: If True, extracts an additional frame at the very end of the video.
            image_format: Image format ('jpg', 'png', 'webp').
            jpeg_quality: Quality for JPEG/WebP (1-100).
            resize: Optional target dimensions (e.g. '1920x1080').
            preserve_resolution: Whether to preserve original resolution if resize is unset.
            progress_callback: Optional callback receiving (current_frame, total_frames).
            
        Returns:
            List of extracted frame record dictionaries.
        """
        video = Path(video_path).resolve()
        out_dir = Path(frames_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if interval_seconds <= 0:
            raise ValueError(f"Frame extraction interval must be a positive number: {interval_seconds}")

        duration = metadata.get("duration_seconds") or 0.0
        ext = image_format.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"

        # Calculate timestamps: 0, interval, 2*interval, ... while t < duration
        timestamps: List[float] = []
        current_t = 0.0

        fps = metadata.get("frame_rate") or 25.0
        frame_dur = 1.0 / fps if fps > 0 else 0.04

        if duration > 0:
            while current_t < duration:
                timestamps.append(round(current_t, 3))
                current_t += interval_seconds

            # End frame extraction: Only if explicitly enabled and not already included
            if include_end_frame:
                end_t = max(0.0, duration - frame_dur)
                if not timestamps or (end_t - timestamps[-1]) >= (0.5 * frame_dur):
                    timestamps.append(round(end_t, 3))
        else:
            # Duration unknown or 0, extract at least frame at 0
            timestamps.append(0.0)

        total_frames = len(timestamps)
        logger.info(
            f"Mode A (Interval): Extracting {total_frames} frame(s) every {interval_seconds}s "
            f"from '{video.name}' (duration: {duration:.2f}s)..."
        )

        filter_scale = build_scale_filter(resize, preserve_resolution)
        codec_args = get_codec_and_quality_args(ext, jpeg_quality)

        extracted_records: List[Dict[str, Any]] = []

        # Determine whether milliseconds are required in filename/CSV
        sub_second = any((ts % 1.0) != 0 for ts in timestamps) or (interval_seconds < 1.0)

        for idx, ts in enumerate(timestamps, start=1):
            ts_str_file = seconds_to_hms(ts, delimiter="-", include_ms=sub_second)
            ts_str_csv = seconds_to_hms(ts, delimiter=":", include_ms=sub_second)

            filename = f"frame_{idx:06d}_{ts_str_file}.{ext}"
            frame_path = out_dir / filename

            # Build accurate FFmpeg extraction command:
            # -ss before -i for fast seek to nearest keyframe, decoding to target timestamp
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-ss", str(ts),
                "-i", str(video),
                "-frames:v", "1",
                "-update", "1",
            ]

            if filter_scale:
                cmd.extend(["-vf", filter_scale])

            cmd.extend(codec_args)
            cmd.append(str(frame_path))

            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error extracting frame {idx} at timestamp {ts}s: {e.stderr}")
                continue

            # Validate extracted image
            is_valid, err_msg, dims = validate_image_cv2(frame_path)
            if not is_valid:
                logger.error(f"Validation failed for {filename}: {err_msg}")
                if frame_path.exists():
                    frame_path.unlink()
                continue

            # Record
            record = {
                "frame_number": idx,
                "timestamp_seconds": int(ts) if not sub_second else round(ts, 3),
                "timestamp": ts_str_csv,
                "filename": filename,
                "width": dims[0] if dims else None,
                "height": dims[1] if dims else None,
            }
            extracted_records.append(record)

            if progress_callback:
                progress_callback(idx, total_frames)

        # Write CSV
        self._write_csv(extracted_records, Path(csv_path))
        logger.info(f"Interval extraction completed: {len(extracted_records)}/{total_frames} frames saved.")
        return extracted_records

    def extract_custom_timestamps(
        self,
        video_path: Path | str,
        frames_dir: Path | str,
        csv_path: Path | str,
        timestamps_input: List[str | float] | str,
        metadata: Dict[str, Any],
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        resize: Optional[str] = None,
        preserve_resolution: bool = True,
    ) -> List[Dict[str, Any]]:
        """Extract frames at explicit custom timestamps (Mode C)."""
        video = Path(video_path).resolve()
        out_dir = Path(frames_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse timestamps input
        raw_list: List[str] = []
        if isinstance(timestamps_input, str):
            # Check if it's a file path
            p = Path(timestamps_input)
            if p.is_file():
                raw_list = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                raw_list = [t.strip() for t in timestamps_input.split(",") if t.strip()]
        elif isinstance(timestamps_input, list):
            raw_list = [str(t).strip() for t in timestamps_input if str(t).strip()]

        parsed_ts: List[float] = []
        for item in raw_list:
            try:
                parsed_ts.append(parse_timestamp(item))
            except ValueError as e:
                logger.warning(f"Skipping invalid timestamp '{item}': {e}")

        if not parsed_ts:
            raise ValueError("No valid timestamps provided for custom timestamp extraction.")

        # Sort chronologically and deduplicate
        parsed_ts = sorted(list(set(parsed_ts)))

        duration = metadata.get("duration_seconds") or 0.0
        ext = image_format.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"

        filter_scale = build_scale_filter(resize, preserve_resolution)
        codec_args = get_codec_and_quality_args(ext, jpeg_quality)

        extracted_records: List[Dict[str, Any]] = []
        sub_second = any((ts % 1.0) != 0 for ts in parsed_ts)

        fps = metadata.get("frame_rate") or 25.0
        frame_dur = 1.0 / fps if fps > 0 else 0.04

        for idx, ts in enumerate(parsed_ts, start=1):
            seek_ts = ts
            if duration > 0 and ts >= duration:
                logger.warning(f"Timestamp {ts}s exceeds or equals video duration ({duration:.2f}s). Clamping to final decodable frame.")
                seek_ts = max(0.0, duration - frame_dur)

            ts_str_file = seconds_to_hms(ts, delimiter="-", include_ms=sub_second)
            ts_str_csv = seconds_to_hms(ts, delimiter=":", include_ms=sub_second)

            filename = f"frame_{idx:06d}_{ts_str_file}.{ext}"
            frame_path = out_dir / filename

            cmd = [
                self.ffmpeg_path,
                "-y",
                "-ss", str(seek_ts),
                "-i", str(video),
                "-frames:v", "1",
                "-update", "1",
            ]
            if filter_scale:
                cmd.extend(["-vf", filter_scale])
            cmd.extend(codec_args)
            cmd.append(str(frame_path))

            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error extracting timestamp {ts}s: {e.stderr}")
                continue

            is_valid, err_msg, dims = validate_image_cv2(frame_path)
            if not is_valid:
                logger.error(f"Validation failed for {filename}: {err_msg}")
                if frame_path.exists():
                    frame_path.unlink()
                continue

            record = {
                "frame_number": idx,
                "timestamp_seconds": int(ts) if not sub_second else round(ts, 3),
                "timestamp": ts_str_csv,
                "filename": filename,
                "width": dims[0] if dims else None,
                "height": dims[1] if dims else None,
            }
            extracted_records.append(record)

        self._write_csv(extracted_records, Path(csv_path))
        logger.info(f"Custom timestamp extraction complete: {len(extracted_records)} frames saved.")
        return extracted_records

    def extract_all_frames(
        self,
        video_path: Path | str,
        frames_dir: Path | str,
        csv_path: Path | str,
        metadata: Dict[str, Any],
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        resize: Optional[str] = None,
        preserve_resolution: bool = True,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract every single decoded frame from the video (Mode B)."""
        video = Path(video_path).resolve()
        out_dir = Path(frames_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        total_frames = metadata.get("total_frame_count") or 0
        fps = metadata.get("frame_rate") or 30.0
        ext = image_format.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"

        # Estimate disk usage: ~150KB per frame for 1080p JPEG, ~1MB for PNG
        est_bytes_per_frame = 200 * 1024 if ext in ("jpg", "webp") else 1024 * 1024
        est_total_bytes = total_frames * est_bytes_per_frame

        logger.warning(
            f"Mode B (All Frames): Video has ~{total_frames} frames at {fps:.2f} FPS. "
            f"Estimated disk usage: ~{format_bytes(est_total_bytes)}. "
            f"Ensure sufficient storage is available!"
        )

        has_space, free_bytes, _ = check_disk_space(out_dir, est_total_bytes)
        if not has_space and not force:
            raise DiskSpaceError(
                f"Insufficient disk space for all-frames extraction: "
                f"Required ~{format_bytes(est_total_bytes)}, available: {format_bytes(free_bytes)}. "
                f"Use --force to proceed anyway."
            )

        filter_scale = build_scale_filter(resize, preserve_resolution)
        codec_args = get_codec_and_quality_args(ext, jpeg_quality)

        output_pattern = str(out_dir / f"frame_%06d.{ext}")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video),
            "-fps_mode", "passthrough",
        ]
        if filter_scale:
            cmd.extend(["-vf", filter_scale])
        cmd.extend(codec_args)
        cmd.append(output_pattern)

        logger.info(f"Executing FFmpeg frame-by-frame streaming extraction...")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(f"FFmpeg all-frames extraction failed: {e.stderr}") from e

        # Read back extracted frames in order, rename to frame_000001_00-00-00-000.ext and record in CSV
        extracted_files = sorted(out_dir.glob(f"frame_*.{ext}"))
        extracted_records: List[Dict[str, Any]] = []

        for idx, fpath in enumerate(extracted_files, start=1):
            ts = (idx - 1) / fps if fps > 0 else float(idx - 1)
            ts_str_file = seconds_to_hms(ts, delimiter="-", include_ms=True)
            ts_str_csv = seconds_to_hms(ts, delimiter=":", include_ms=True)
            new_filename = f"frame_{idx:06d}_{ts_str_file}.{ext}"
            new_path = out_dir / new_filename

            if new_path != fpath:
                fpath.rename(new_path)

            record = {
                "frame_number": idx,
                "timestamp_seconds": round(ts, 3),
                "timestamp": ts_str_csv,
                "filename": new_filename,
            }
            extracted_records.append(record)

        self._write_csv(extracted_records, Path(csv_path))
        logger.info(f"All-frames extraction complete: {len(extracted_records)} frames saved.")
        return extracted_records

    def extract_scene_changes(
        self,
        video_path: Path | str,
        frames_dir: Path | str,
        csv_path: Path | str,
        metadata: Dict[str, Any],
        scene_threshold: float = 0.3,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
        resize: Optional[str] = None,
        preserve_resolution: bool = True,
    ) -> List[Dict[str, Any]]:
        """Extract representative frames when significant scene transitions occur (Mode D)."""
        video = Path(video_path).resolve()
        out_dir = Path(frames_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = image_format.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"

        logger.info(f"Mode D (Scene): Detecting scene changes with threshold {scene_threshold}...")

        # Build FFmpeg filter: select='gt(scene,THRESHOLD)',showinfo
        vf_parts = [f"select='gt(scene,{scene_threshold})'", "showinfo"]
        scale_f = build_scale_filter(resize, preserve_resolution)
        if scale_f:
            vf_parts.append(scale_f)

        filter_str = ",".join(vf_parts)
        codec_args = get_codec_and_quality_args(ext, jpeg_quality)
        temp_pattern = str(out_dir / f"scene_raw_%06d.{ext}")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video),
            "-vf", filter_str,
            "-fps_mode", "vfr",
        ]
        cmd.extend(codec_args)
        cmd.append(temp_pattern)

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            stderr_out = res.stderr
        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(f"FFmpeg scene detection failed: {e.stderr}") from e

        # Parse timestamps from showinfo lines in stderr:
        # e.g.: [Parsed_showinfo_1 @ 0x...] n:   0 pts:  38400 pts_time:3 ...
        pts_times: List[float] = []
        for line in stderr_out.splitlines():
            if "Parsed_showinfo" in line and "pts_time:" in line:
                m = re.search(r"pts_time:([0-9.]+)", line)
                if m:
                    try:
                        pts_times.append(float(m.group(1)))
                    except ValueError:
                        pass

        # Sort and associate with generated files
        raw_files = sorted(out_dir.glob(f"scene_raw_*.{ext}"))
        extracted_records: List[Dict[str, Any]] = []

        # If no scenes were detected, also extract the first frame at 00:00:00
        if not raw_files:
            logger.info("No scene changes exceeded threshold. Extracting initial frame at 00:00:00.")
            first_frame_path = out_dir / f"frame_000001_00-00-00.{ext}"
            cmd_first = [
                self.ffmpeg_path,
                "-y",
                "-ss", "0",
                "-i", str(video),
                "-frames:v", "1",
                "-update", "1",
            ]
            if scale_f:
                cmd_first.extend(["-vf", scale_f])
            cmd_first.extend(codec_args)
            cmd_first.append(str(first_frame_path))
            subprocess.run(cmd_first, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if first_frame_path.is_file() and validate_image_cv2(first_frame_path)[0]:
                extracted_records.append({
                    "frame_number": 1,
                    "timestamp_seconds": 0,
                    "timestamp": "00:00:00",
                    "filename": first_frame_path.name,
                })
        else:
            for idx, fpath in enumerate(raw_files, start=1):
                ts = pts_times[idx - 1] if idx - 1 < len(pts_times) else 0.0
                ts_str_file = seconds_to_hms(ts, delimiter="-", include_ms=True)
                ts_str_csv = seconds_to_hms(ts, delimiter=":", include_ms=True)

                final_name = f"frame_{idx:06d}_{ts_str_file}.{ext}"
                final_path = out_dir / final_name
                fpath.rename(final_path)

                is_valid, err_msg, dims = validate_image_cv2(final_path)
                if not is_valid:
                    logger.error(f"Validation failed for scene frame {final_name}: {err_msg}")
                    if final_path.exists():
                        final_path.unlink()
                    continue

                extracted_records.append({
                    "frame_number": idx,
                    "timestamp_seconds": round(ts, 3),
                    "timestamp": ts_str_csv,
                    "filename": final_name,
                })

        self._write_csv(extracted_records, Path(csv_path))
        logger.info(f"Scene detection extraction complete: {len(extracted_records)} frames saved.")
        return extracted_records

