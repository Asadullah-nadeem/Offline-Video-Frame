#!/usr/bin/env python3
"""Main Command-Line Interface for the Offline Video Frame Extraction and Audio Transcription System."""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to sys.path so scripts can be imported from root
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from scripts.file_manager import find_video_files
from scripts.utils import FFmpegNotFoundError, setup_logger, verify_ffmpeg
from scripts.video_processor import VideoProcessor

logger = setup_logger("VideoProcessor.Main")


def load_config(config_path: Path | str) -> Dict[str, Any]:
    """Load configuration JSON file with sensible fallback defaults."""
    default_config = {
        "input_directory": "./input",
        "output_directory": "./output",
        "extraction_mode": "interval",
        "frame_interval_seconds": 60,
        "image_format": "jpg",
        "jpeg_quality": 95,
        "preserve_resolution": True,
        "extract_audio": True,
        "enable_transcription": False,
        "transcription_language": "en",
        "transcription_model_path": "./models/vosk-en",
        "save_metadata": True,
        "generate_csv": True,
        "generate_srt": True,
        "recursive_scan": False,
        "max_parallel_jobs": 2,
        "overwrite_existing": False,
    }

    cfg_file = Path(config_path).resolve()
    if not cfg_file.is_file():
        logger.warning(f"Configuration file not found at '{cfg_file}'. Using internal defaults.")
        return default_config

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        default_config.update(user_config)
        return default_config
    except Exception as e:
        logger.error(f"Error reading configuration file '{cfg_file}': {e}. Using defaults.")
        return default_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Offline Video Frame Extraction and Audio Transcription System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        dest="input_path",
        type=str,
        default=None,
        help="Path to an individual video file or a directory containing videos.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        type=str,
        default=None,
        help="Destination directory for output folders.",
    )
    parser.add_argument(
        "--config", "-c",
        dest="config_path",
        type=str,
        default="config/settings.json",
        help="Path to configuration JSON file.",
    )
    parser.add_argument(
        "--mode", "-m",
        dest="mode",
        choices=["interval", "all-frames", "timestamps", "scene"],
        default=None,
        help="Frame extraction mode.",
    )
    parser.add_argument(
        "--interval",
        dest="interval",
        type=float,
        default=None,
        help="Frame extraction interval in seconds for 'interval' mode (e.g. 1, 5, 10, 60).",
    )
    parser.add_argument(
        "--timestamps",
        dest="timestamps",
        type=str,
        default=None,
        help="Comma-separated timestamps (e.g. '00:00:15,00:01:30') or path to file for 'timestamps' mode.",
    )
    parser.add_argument(
        "--scene-threshold",
        dest="scene_threshold",
        type=float,
        default=None,
        help="Scene change detection sensitivity (0.0 to 1.0) for 'scene' mode.",
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        choices=["jpg", "jpeg", "png", "webp"],
        default=None,
        help="Image format for extracted frames.",
    )
    parser.add_argument(
        "--quality",
        dest="jpeg_quality",
        type=int,
        default=None,
        help="JPEG/WebP quality percentage (1-100).",
    )
    parser.add_argument(
        "--resize",
        dest="resize",
        type=str,
        default=None,
        help="Resize frames to dimensions (e.g. '1920x1080', '1280x720') preserving aspect ratio.",
    )
    parser.add_argument(
        "--include-end-frame",
        dest="include_end_frame",
        action="store_true",
        help="In interval mode, extract an additional frame at the very end of the video duration.",
    )
    parser.add_argument(
        "--extract-audio",
        dest="extract_audio",
        action="store_true",
        default=None,
        help="Enable audio extraction to 16kHz mono WAV.",
    )
    parser.add_argument(
        "--no-audio",
        dest="no_audio",
        action="store_true",
        help="Disable audio extraction entirely.",
    )
    parser.add_argument(
        "--preserve-original-audio",
        dest="preserve_original_audio",
        action="store_true",
        help="Preserve an exact copy of the source audio stream in its original codec.",
    )
    parser.add_argument(
        "--audio-stream-index",
        dest="audio_stream_index",
        type=int,
        default=None,
        help="Zero-based index of audio stream to extract if multiple streams are present.",
    )
    parser.add_argument(
        "--transcribe",
        dest="transcribe",
        action="store_true",
        default=None,
        help="Enable offline speech-to-text recognition using Vosk.",
    )
    parser.add_argument(
        "--language",
        dest="language",
        type=str,
        default=None,
        help="Spoken language code for transcription (e.g. 'en', 'hi', 'ur').",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default=None,
        help="Path to local offline Vosk acoustic model directory.",
    )
    parser.add_argument(
        "--recursive", "-r",
        dest="recursive",
        action="store_true",
        default=None,
        help="Scan input directory recursively for video files.",
    )
    parser.add_argument(
        "--workers", "-w",
        dest="workers",
        type=int,
        default=None,
        help="Maximum simultaneous parallel jobs for batch processing.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing output directory instead of creating a numbered variant.",
    )
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Force operation even if disk space warnings are triggered.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.",
    )

    return parser.parse_args()


def build_final_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge configuration file and command-line arguments."""
    config = load_config(args.config_path)

    # CLI overrides
    if args.input_path is not None:
        config["input_path"] = args.input_path
    elif "input_path" not in config:
        config["input_path"] = config.get("input_directory", "./input")

    if args.output_dir is not None:
        config["output_directory"] = args.output_dir

    if args.mode is not None:
        config["extraction_mode"] = args.mode

    if args.interval is not None:
        config["frame_interval_seconds"] = args.interval

    if args.timestamps is not None:
        config["timestamps"] = args.timestamps

    if args.scene_threshold is not None:
        config["scene_threshold"] = args.scene_threshold

    if args.image_format is not None:
        config["image_format"] = args.image_format

    if args.jpeg_quality is not None:
        config["jpeg_quality"] = args.jpeg_quality

    if args.resize is not None:
        config["resize"] = args.resize

    if args.include_end_frame:
        config["include_end_frame"] = True

    if args.no_audio:
        config["extract_audio"] = False
    elif args.extract_audio is not None:
        config["extract_audio"] = True

    if args.preserve_original_audio:
        config["preserve_original_audio"] = True

    if args.audio_stream_index is not None:
        config["audio_stream_index"] = args.audio_stream_index

    if args.transcribe is not None:
        config["enable_transcription"] = args.transcribe

    if args.language is not None:
        config["transcription_language"] = args.language

    if args.model_path is not None:
        config["transcription_model_path"] = args.model_path

    if args.recursive is not None:
        config["recursive_scan"] = args.recursive

    if args.workers is not None:
        config["max_parallel_jobs"] = args.workers

    if args.overwrite is not None:
        config["overwrite_existing"] = args.overwrite

    if args.force:
        config["force"] = True

    config["log_level"] = args.log_level
    return config


def main() -> int:
    """CLI execution entrypoint."""
    args = parse_arguments()
    config = build_final_config(args)

    logger.setLevel(getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO))

    # 1. Verify FFmpeg installation
    try:
        ffmpeg_bin, ffprobe_bin = verify_ffmpeg()
        logger.debug(f"Verified FFmpeg: {ffmpeg_bin}, FFprobe: {ffprobe_bin}")
    except FFmpegNotFoundError as e:
        logger.error(str(e))
        return 1

    # 2. Discover video files
    input_target = config.get("input_path")
    if not input_target:
        logger.error("No input video file or directory specified. Use --input <path>.")
        return 1

    try:
        video_files = find_video_files(
            input_path=input_target,
            recursive=config.get("recursive_scan", False),
        )
    except Exception as e:
        logger.error(f"Failed to find input video files: {e}")
        return 1

    if not video_files:
        logger.warning(f"No supported video files found at '{input_target}'. Exiting.")
        return 0

    logger.info(f"Discovered {len(video_files)} video file(s) for processing.")

    # 3. Process video(s)
    processor = VideoProcessor(config)
    max_workers = max(1, int(config.get("max_parallel_jobs", 1)))
    results: List[Dict[str, Any]] = []

    start_batch_time = time.time()

    if len(video_files) == 1 or max_workers == 1:
        # Sequential execution
        for vfile in video_files:
            logger.info(f"Processing: {vfile.name}")
            res = processor.process(vfile)
            results.append(res)
    else:
        # Parallel execution with thread pool (each thread invokes FFmpeg subprocesses)
        logger.info(f"Running batch processing with {max_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(processor.process, vfile): vfile for vfile in video_files}
            for future in as_completed(futures):
                vfile = futures[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(f"Unexpected worker exception processing '{vfile.name}': {e}")
                    results.append({
                        "video_name": vfile.name,
                        "status": "FAILED",
                        "error": str(e),
                        "total_frames_extracted": 0,
                    })

    total_batch_duration = round(time.time() - start_batch_time, 2)

    # 4. Final summary report
    print("\n" + "=" * 80)
    print("                     VIDEO PROCESSING BATCH SUMMARY")
    print("=" * 80)
    print(f"{'Video Filename':<30} | {'Status':<10} | {'Frames':<8} | {'Audio':<12} | {'Time (s)':<8}")
    print("-" * 80)

    success_count = 0
    failure_count = 0
    total_frames = 0

    for r in results:
        vname = r.get("video_name", "Unknown")
        if len(vname) > 28:
            vname = vname[:25] + "..."
        status = r.get("status", "UNKNOWN")
        frames = r.get("total_frames_extracted", 0)
        audio = r.get("audio_status", "N/A")
        dur = r.get("duration_seconds", 0.0)

        if status == "SUCCESS":
            success_count += 1
            total_frames += frames
        else:
            failure_count += 1

        print(f"{vname:<30} | {status:<10} | {frames:<8} | {audio:<12} | {dur:<8.2f}")

    print("-" * 80)
    print(f"Total Videos Processed : {len(results)}")
    print(f"Successful Extractions : {success_count}")
    print(f"Failed Extractions     : {failure_count}")
    print(f"Total Frames Extracted : {total_frames}")
    print(f"Total Batch Duration   : {total_batch_duration}s")
    print("=" * 80 + "\n")

    return 0 if (success_count > 0 or len(results) == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
