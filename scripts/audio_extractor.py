"""Audio extraction and format conversion using FFmpeg."""

import logging
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.utils import VideoProcessingError, verify_ffmpeg

logger = logging.getLogger("VideoProcessor.AudioExtractor")

CODEC_EXTENSION_MAP = {
    "aac": "m4a",
    "mp3": "mp3",
    "opus": "opus",
    "flac": "flac",
    "vorbis": "ogg",
    "pcm_s16le": "wav",
    "pcm_s24le": "wav",
    "ac3": "ac3",
    "eac3": "eac3",
    "wmav2": "wma",
}


class AudioExtractor:
    """Handles audio detection, conversion to 16kHz mono WAV, and optional original audio preservation."""

    def __init__(self, ffmpeg_path: Optional[str] = None):
        if ffmpeg_path:
            self.ffmpeg_path = ffmpeg_path
        else:
            self.ffmpeg_path, _ = verify_ffmpeg()

    def extract(
        self,
        video_path: Path | str,
        output_audio_dir: Path | str,
        metadata: Dict[str, Any],
        stream_index: int = 0,
        preserve_original: bool = False,
    ) -> Dict[str, Optional[Path]]:
        """Extract audio from video.
        
        Args:
            video_path: Path to the input video.
            output_audio_dir: Directory where extracted audio files will be saved.
            metadata: Video metadata dictionary retrieved from MetadataExtractor.
            stream_index: Zero-based index of the audio stream to extract.
            preserve_original: If True, also extracts a copy of the stream in original codec.
            
        Returns:
            Dict with keys:
              - 'wav': Path to 16kHz mono WAV file or None.
              - 'original': Path to original-codec audio file or None.
              - 'status': 'EXTRACTED', 'SKIPPED_NO_AUDIO', or 'FAILED'.
        """
        video = Path(video_path).resolve()
        out_dir = Path(output_audio_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_count = metadata.get("audio_stream_count", 0)
        if audio_count == 0:
            logger.info(f"No audio stream found in '{video.name}'. Skipping audio extraction.")
            return {"wav": None, "original": None, "status": "SKIPPED_NO_AUDIO"}

        # Validate stream index
        audio_streams = metadata.get("audio_streams", [])
        if stream_index < 0 or stream_index >= len(audio_streams):
            logger.warning(
                f"Requested audio stream index {stream_index} is out of bounds (found {len(audio_streams)} streams). "
                f"Defaulting to stream index 0."
            )
            stream_index = 0

        target_stream = audio_streams[stream_index] if audio_streams else {}
        stream_codec = target_stream.get("codec") or "unknown"
        logger.info(f"Extracting audio stream #{stream_index} ({stream_codec}) from '{video.name}'...")

        wav_path = out_dir / "extracted_audio.wav"

        # FFmpeg command for 16kHz mono PCM WAV
        cmd_wav = [
            self.ffmpeg_path,
            "-y",
            "-i", str(video),
            "-map", f"0:a:{stream_index}",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            str(wav_path),
        ]

        try:
            res = subprocess.run(cmd_wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract 16kHz WAV from '{video.name}': {e.stderr}")
            return {"wav": None, "original": None, "status": "FAILED"}

        # Validate extracted WAV file
        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            logger.error(f"Extracted WAV file is empty or missing: {wav_path}")
            return {"wav": None, "original": None, "status": "FAILED"}

        try:
            with wave.open(str(wav_path), "rb") as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                frames = wf.getnframes()
                logger.debug(f"Validated WAV: {channels} channel(s), {rate} Hz, {frames} frames.")
        except Exception as e:
            logger.error(f"Extracted WAV validation failed: {e}")
            return {"wav": None, "original": None, "status": "FAILED"}

        original_path: Optional[Path] = None
        if preserve_original:
            ext = CODEC_EXTENSION_MAP.get(stream_codec.lower(), "mka")
            original_path = out_dir / f"original_audio.{ext}"
            cmd_orig = [
                self.ffmpeg_path,
                "-y",
                "-i", str(video),
                "-map", f"0:a:{stream_index}",
                "-vn",
                "-c:a", "copy",
                str(original_path),
            ]
            try:
                subprocess.run(cmd_orig, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                if original_path.is_file() and original_path.stat().st_size > 0:
                    logger.info(f"Saved original-quality audio to: {original_path.name}")
                else:
                    logger.warning(f"Original audio file was empty or missing: {original_path}")
                    original_path = None
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to copy original audio stream directly: {e.stderr}")
                original_path = None

        logger.info(f"Successfully extracted audio to: {wav_path.name}")
        return {
            "wav": wav_path,
            "original": original_path,
            "status": "EXTRACTED",
        }

