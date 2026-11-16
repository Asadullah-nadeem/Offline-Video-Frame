"""Orchestrator for processing a single video file through metadata, frame, audio, and transcription pipelines."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from scripts.audio_extractor import AudioExtractor
from scripts.file_manager import create_output_structure
from scripts.frame_extractor import FrameExtractor
from scripts.metadata_extractor import MetadataExtractor
from scripts.transcription import OfflineTranscriber
from scripts.utils import VideoProcessingError, setup_logger


class VideoProcessor:
    """Processes an individual video through all extraction and transcription stages."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata_extractor = MetadataExtractor()
        self.frame_extractor = FrameExtractor()
        self.audio_extractor = AudioExtractor()

    def process(self, video_path: Path | str) -> Dict[str, Any]:
        """Process a single video file according to configuration.
        
        Args:
            video_path: Path to the target video file.
            
        Returns:
            Dictionary containing processing summary and output artifact paths.
        """
        start_time = time.time()
        video = Path(video_path).resolve()

        base_output_dir = Path(self.config.get("output_directory", "./output"))
        overwrite = self.config.get("overwrite_existing", False)

        # 1. Automatic folder creation
        video_out_dir, subdirs = create_output_structure(base_output_dir, video, overwrite=overwrite)

        # 2. Configure dedicated per-video logger
        video_log_path = subdirs["logs"] / "processing.log"
        logger = setup_logger(
            name=f"Processor_{video.stem}",
            log_file=video_log_path,
            log_level=self.config.get("log_level", "INFO"),
        )

        logger.info(f"Starting processing pipeline for '{video.name}'")
        logger.info(f"Dedicated output directory: {video_out_dir}")

        metadata_json_path = subdirs["metadata"] / "video_info.json"
        csv_path = subdirs["metadata"] / "frames.csv"

        metadata: Dict[str, Any] = {}
        extracted_frames = []
        audio_result = {"wav": None, "original": None, "status": "NOT_RUN"}
        transcription_result = {"status": "NOT_RUN"}

        try:
            # 3. Retrieve video metadata
            logger.info("Extracting video metadata...")
            metadata = self.metadata_extractor.inspect(video)
            logger.info(
                f"Video format: {metadata.get('container_format')}, codec: {metadata.get('video_codec')}, "
                f"resolution: {metadata.get('width')}x{metadata.get('height')}, "
                f"duration: {metadata.get('duration_formatted')} ({metadata.get('duration_seconds')}s), "
                f"audio streams: {metadata.get('audio_stream_count')}"
            )

            # Record initial metadata
            self.metadata_extractor.save(metadata, metadata_json_path)

            # 4. Frame extraction
            mode = self.config.get("extraction_mode", "interval")
            img_format = self.config.get("image_format", "jpg")
            jpeg_quality = self.config.get("jpeg_quality", 95)
            resize = self.config.get("resize")
            preserve_res = self.config.get("preserve_resolution", True)
            interval_sec = float(self.config.get("frame_interval_seconds", 60))
            include_end_frame = self.config.get("include_end_frame", False)

            metadata["frame_extraction_interval"] = interval_sec if mode == "interval" else mode

            if mode == "interval":
                extracted_frames = self.frame_extractor.extract_interval(
                    video_path=video,
                    frames_dir=subdirs["frames"],
                    csv_path=csv_path,
                    metadata=metadata,
                    interval_seconds=interval_sec,
                    include_end_frame=include_end_frame,
                    image_format=img_format,
                    jpeg_quality=jpeg_quality,
                    resize=resize,
                    preserve_resolution=preserve_res,
                )
            elif mode == "all-frames":
                extracted_frames = self.frame_extractor.extract_all_frames(
                    video_path=video,
                    frames_dir=subdirs["frames"],
                    csv_path=csv_path,
                    metadata=metadata,
                    image_format=img_format,
                    jpeg_quality=jpeg_quality,
                    resize=resize,
                    preserve_resolution=preserve_res,
                    force=self.config.get("force", False),
                )
            elif mode == "timestamps":
                ts_input = self.config.get("timestamps", [])
                extracted_frames = self.frame_extractor.extract_custom_timestamps(
                    video_path=video,
                    frames_dir=subdirs["frames"],
                    csv_path=csv_path,
                    timestamps_input=ts_input,
                    metadata=metadata,
                    image_format=img_format,
                    jpeg_quality=jpeg_quality,
                    resize=resize,
                    preserve_resolution=preserve_res,
                )
            elif mode == "scene":
                scene_threshold = float(self.config.get("scene_threshold", 0.3))
                extracted_frames = self.frame_extractor.extract_scene_changes(
                    video_path=video,
                    frames_dir=subdirs["frames"],
                    csv_path=csv_path,
                    metadata=metadata,
                    scene_threshold=scene_threshold,
                    image_format=img_format,
                    jpeg_quality=jpeg_quality,
                    resize=resize,
                    preserve_resolution=preserve_res,
                )
            else:
                raise ValueError(f"Unknown frame extraction mode: {mode}")

            # 5. Audio extraction
            if self.config.get("extract_audio", True):
                stream_idx = int(self.config.get("audio_stream_index", 0))
                preserve_orig = self.config.get("preserve_original_audio", False)
                audio_result = self.audio_extractor.extract(
                    video_path=video,
                    output_audio_dir=subdirs["audio"],
                    metadata=metadata,
                    stream_index=stream_idx,
                    preserve_original=preserve_orig,
                )

            # 6. Offline speech transcription (optional)
            if self.config.get("enable_transcription", False):
                wav_file = audio_result.get("wav")
                if wav_file and wav_file.is_file():
                    model_path = self.config.get("transcription_model_path", "./models/vosk-en")
                    lang = self.config.get("transcription_language", "en")
                    transcriber = OfflineTranscriber(model_path=model_path, language=lang)
                    transcription_result = transcriber.transcribe(
                        wav_path=wav_file,
                        output_dir=subdirs["transcript"],
                    )
                else:
                    logger.warning("Transcription enabled, but no extracted WAV audio is available.")
                    transcription_result = {"status": "SKIPPED_NO_AUDIO"}

            elapsed = round(time.time() - start_time, 2)
            metadata["total_extracted_images"] = len(extracted_frames)
            metadata["processing_duration_seconds"] = elapsed
            metadata["processing_status"] = "SUCCESS"

            # Final save of metadata
            self.metadata_extractor.save(metadata, metadata_json_path)
            logger.info(f"Video processing finished successfully in {elapsed}s.")

            return {
                "video_name": video.name,
                "video_path": str(video),
                "status": "SUCCESS",
                "output_dir": str(video_out_dir),
                "total_frames_extracted": len(extracted_frames),
                "audio_status": audio_result.get("status"),
                "transcription_status": transcription_result.get("status"),
                "duration_seconds": elapsed,
            }

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Error processing video '{video.name}': {e}", exc_info=True)
            
            fail_metadata = metadata if metadata else {
                "original_filename": video.name,
                "file_path": str(video),
                "container_format": None,
                "duration_seconds": None,
                "duration_formatted": None,
                "width": None,
                "height": None,
                "video_codec": None,
                "audio_codec": None,
                "audio_stream_count": 0,
                "total_frame_count": None,
                "frame_rate": None,
                "display_aspect_ratio": None,
                "creation_timestamp": None,
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "frame_extraction_interval": self.config.get("extraction_mode", "interval"),
            }
            fail_metadata["processing_duration_seconds"] = elapsed
            fail_metadata["processing_status"] = "FAILED"
            fail_metadata["error_message"] = str(e)
            fail_metadata["total_extracted_images"] = len(extracted_frames)
            
            try:
                self.metadata_extractor.save(fail_metadata, metadata_json_path)
            except Exception as save_err:
                logger.error(f"Failed to write failure metadata to {metadata_json_path}: {save_err}")

            return {
                "video_name": video.name,
                "video_path": str(video),
                "status": "FAILED",
                "error": str(e),
                "output_dir": str(video_out_dir),
                "total_frames_extracted": len(extracted_frames),
                "duration_seconds": elapsed,
            }

