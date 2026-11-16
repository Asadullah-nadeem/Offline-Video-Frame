"""Offline speech-to-text transcription using Vosk."""

import json
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.utils import seconds_to_hms

logger = logging.getLogger("VideoProcessor.Transcription")


def format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_millis = round(seconds * 1000)
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class OfflineTranscriber:
    """Transcribes audio using locally installed Vosk acoustic models without internet connectivity."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        language: str = "en",
    ):
        self.language = language
        self.model_path = Path(model_path).resolve() if model_path else None
        self.vosk_available = False
        self._model = None

        # Check Vosk installation
        try:
            import vosk
            from vosk import SetLogLevel
            SetLogLevel(-1)  # Suppress verbose Kaldi C++ engine logs
            self.vosk_available = True
        except ImportError:
            logger.warning(
                "Vosk library is not installed. Offline speech transcription will be unavailable. "
                "Install it using: pip install vosk"
            )

    def is_available(self) -> tuple[bool, str]:
        """Check if both Vosk library and local model directory are valid.
        
        Returns:
            (is_ready, message)
        """
        if not self.vosk_available:
            return False, "Vosk Python library is not installed."

        if not self.model_path:
            return False, "No Vosk model directory was specified."

        if not self.model_path.exists():
            return False, f"Vosk model path does not exist: {self.model_path}"

        if not self.model_path.is_dir():
            return False, f"Vosk model path is not a directory: {self.model_path}"

        # Vosk models typically contain am/ or am.bin, conf/ or graph/
        sub_items = [p.name for p in self.model_path.iterdir()]
        if not sub_items:
            return False, f"Vosk model directory is empty: {self.model_path}"

        return True, "Model is ready."

    def _load_model(self):
        """Lazy load the Vosk model from the local directory."""
        if self._model is not None:
            return self._model

        import vosk
        logger.info(f"Loading local offline Vosk model from: {self.model_path}")
        # Note: We strictly pass model_path to ensure NO network downloads occur
        self._model = vosk.Model(str(self.model_path))
        return self._model

    def transcribe(
        self,
        wav_path: Path | str,
        output_dir: Path | str,
    ) -> Dict[str, Any]:
        """Transcribe a 16kHz mono WAV file and generate TXT, JSON, and SRT transcripts.
        
        Args:
            wav_path: Path to the 16kHz mono PCM WAV file.
            output_dir: Directory where transcript files will be saved.
            
        Returns:
            Dict containing transcription results and file paths.
        """
        wav_file = Path(wav_path).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_path = out_dir / "transcript.txt"
        json_path = out_dir / "transcript.json"
        srt_path = out_dir / "transcript.srt"

        ready, reason = self.is_available()
        if not ready:
            logger.warning(f"Skipping transcription: {reason}")
            # Create informative placeholder files
            txt_path.write_text(f"[Transcription skipped: {reason}]\n", encoding="utf-8")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"status": "SKIPPED", "reason": reason, "segments": []}, f, indent=2)
            srt_path.write_text("", encoding="utf-8")
            return {
                "status": "SKIPPED",
                "reason": reason,
                "txt": txt_path,
                "json": json_path,
                "srt": srt_path,
                "segments_count": 0,
            }

        if not wav_file.is_file() or wav_file.stat().st_size == 0:
            msg = "WAV file is missing or empty."
            logger.warning(f"Cannot transcribe: {msg}")
            return {"status": "FAILED", "reason": msg}

        try:
            model = self._load_model()
            import vosk
        except Exception as e:
            err = f"Failed to initialize Vosk model: {e}"
            logger.error(err)
            return {"status": "FAILED", "reason": err}

        logger.info(f"Transcribing '{wav_file.name}' using offline Vosk model...")

        try:
            with wave.open(str(wav_file), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                rec = vosk.KaldiRecognizer(model, sample_rate)
                rec.SetWords(True)

                raw_segments: List[Dict[str, Any]] = []

                # Chunk-based streaming read: 4000 frames at a time
                chunk_size = 4000
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        if res.get("text"):
                            raw_segments.append(res)

                final_res = json.loads(rec.FinalResult())
                if final_res.get("text"):
                    raw_segments.append(final_res)
        except Exception as e:
            err = f"Error during audio decoding/recognition: {e}"
            logger.error(err)
            return {"status": "FAILED", "reason": err}

        # Process segments and timestamps
        structured_segments = []
        txt_lines = []
        srt_entries = []
        full_text_parts = []

        srt_index = 1
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            # Capitalize first letter of segment for readability
            formatted_text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            full_text_parts.append(formatted_text)

            words = seg.get("result", [])
            start_sec = words[0]["start"] if words else 0.0
            end_sec = words[-1]["end"] if words else start_sec + 2.0

            start_hms = seconds_to_hms(start_sec, delimiter=":")
            end_hms = seconds_to_hms(end_sec, delimiter=":")

            # Text format: [HH:MM:SS] Text...
            txt_lines.append(f"[{start_hms}] {formatted_text}.\n")

            # SRT format
            srt_start = format_srt_time(start_sec)
            srt_end = format_srt_time(end_sec)
            srt_entries.append(f"{srt_index}\n{srt_start} --> {srt_end}\n{formatted_text}.\n\n")
            srt_index += 1

            structured_segments.append({
                "segment_index": len(structured_segments) + 1,
                "start_seconds": round(start_sec, 3),
                "end_seconds": round(end_sec, 3),
                "start_formatted": start_hms,
                "end_formatted": end_hms,
                "text": formatted_text,
                "words": words,
            })

        # Handle empty/silent audio
        if not structured_segments:
            logger.info(f"No recognizable speech detected in '{wav_file.name}'.")
            txt_content = "[No recognizable speech detected]\n"
            json_data = {
                "audio_file": wav_file.name,
                "language": self.language,
                "model_path": str(self.model_path),
                "segments_count": 0,
                "full_text": "",
                "segments": [],
                "status": "SUCCESS_NO_SPEECH",
            }
            srt_content = ""
        else:
            txt_content = "\n".join(txt_lines) + "\n"
            json_data = {
                "audio_file": wav_file.name,
                "language": self.language,
                "model_path": str(self.model_path),
                "segments_count": len(structured_segments),
                "full_text": ". ".join(full_text_parts) + ".",
                "segments": structured_segments,
                "status": "SUCCESS",
            }
            srt_content = "".join(srt_entries)

        # Write output files
        txt_path.write_text(txt_content, encoding="utf-8")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        srt_path.write_text(srt_content, encoding="utf-8")

        logger.info(
            f"Transcription complete. Saved: {txt_path.name}, {json_path.name}, {srt_path.name} "
            f"({len(structured_segments)} segments recognized)."
        )

        return {
            "status": json_data["status"],
            "txt": txt_path,
            "json": json_path,
            "srt": srt_path,
            "segments_count": len(structured_segments),
            "full_text": json_data["full_text"],
        }

