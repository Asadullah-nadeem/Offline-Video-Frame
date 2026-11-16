"""Comprehensive automated test suite covering all 14 video processing scenarios."""

import csv
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

# Add project root to sys.path
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.metadata_extractor import MetadataExtractor
from scripts.video_processor import VideoProcessor
from scripts.utils import verify_ffmpeg, CorruptedVideoError


class TestVideoProcessingSystem(unittest.TestCase):
    """Test suite executing all 14 mandatory test cases."""

    @classmethod
    def setUpClass(cls):
        cls.ffmpeg_bin, cls.ffprobe_bin = verify_ffmpeg()
        cls.test_dir = project_root / "tests" / "test_media"
        cls.output_dir = project_root / "tests" / "test_outputs"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up test outputs and generated media if needed
        pass

    def _generate_video(
        self,
        output_filename: str,
        duration: int = 5,
        fps: int = 25,
        width: int = 320,
        height: int = 240,
        audio_frequency: int = 1000,
        audio_streams: int = 1,
        audio_wav: Path | None = None,
        vfr: bool = False,
    ) -> Path:
        """Helper to synthesize valid test video files using FFmpeg."""
        out_path = self.test_dir / output_filename
        if out_path.exists():
            return out_path

        cmd = [self.ffmpeg_bin, "-y"]

        # Video source
        if vfr:
            # Variable frame rate via mpdecimate filter
            cmd.extend([
                "-f", "lavfi",
                "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
            ])
        else:
            cmd.extend([
                "-f", "lavfi",
                "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
            ])

        # Audio source
        if audio_wav and audio_wav.is_file():
            cmd.extend(["-i", str(audio_wav)])
            cmd.extend(["-map", "0:v", "-map", "1:a", "-shortest"])
        elif audio_streams == 0:
            cmd.append("-an")
        elif audio_streams == 1:
            cmd.extend([
                "-f", "lavfi",
                "-i", f"sine=frequency={audio_frequency}:duration={duration}",
                "-map", "0:v", "-map", "1:a",
            ])
        elif audio_streams > 1:
            for i in range(audio_streams):
                freq = audio_frequency * (i + 1)
                cmd.extend([
                    "-f", "lavfi",
                    "-i", f"sine=frequency={freq}:duration={duration}",
                ])
            cmd.extend(["-map", "0:v"])
            for i in range(audio_streams):
                cmd.extend(["-map", f"{i + 1}:a"])

        if vfr:
            cmd.extend(["-vf", "mpdecimate", "-fps_mode", "vfr"])

        cmd.extend(["-c:v", "libx264", "-preset", "ultrafast"])
        if audio_streams > 0 or audio_wav:
            cmd.extend(["-c:a", "aac", "-b:a", "64k"])

        if duration > 100:
            # Fast encoding for long videos
            cmd.extend(["-g", "50", "-threads", "4"])

        cmd.append(str(out_path))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out_path

    # -------------------------------------------------------------------------
    # Test 1: 10-second video
    # -------------------------------------------------------------------------
    def test_01_ten_second_video(self):
        """Test processing of a 10-second video with 2s interval."""
        vfile = self._generate_video("test_01_10s.mp4", duration=10)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "extract_audio": True,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        # 10s video with 2s interval -> timestamps: 0, 2, 4, 6, 8 (5 frames)
        self.assertEqual(res["total_frames_extracted"], 5)

        out_path = Path(res["output_dir"])
        self.assertTrue((out_path / "metadata" / "video_info.json").exists())
        self.assertTrue((out_path / "metadata" / "frames.csv").exists())
        self.assertTrue((out_path / "audio" / "extracted_audio.wav").exists())
        self.assertEqual(len(list((out_path / "frames").glob("*.jpg"))), 5)

    # -------------------------------------------------------------------------
    # Test 2: 1-minute video
    # -------------------------------------------------------------------------
    def test_02_one_minute_video(self):
        """Test processing of a 1-minute (60s) video with 10s interval."""
        vfile = self._generate_video("test_02_1m.mp4", duration=60)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 10,
            "extract_audio": True,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        # 60s video with 10s interval -> timestamps: 0, 10, 20, 30, 40, 50 (6 frames)
        self.assertEqual(res["total_frames_extracted"], 6)

        out_path = Path(res["output_dir"])
        frames = sorted(list((out_path / "frames").glob("*.jpg")))
        self.assertEqual(len(frames), 6)
        # Verify first and last filenames
        self.assertTrue(frames[0].name.startswith("frame_000001_00-00-00"))
        self.assertTrue(frames[5].name.startswith("frame_000006_00-00-50"))

    # -------------------------------------------------------------------------
    # Test 3: 10-minute video (Primary Expectation: Exactly 10 frames at 60s interval)
    # -------------------------------------------------------------------------
    def test_03_ten_minute_video_exact_ten_frames(self):
        """Primary Expectation: 600s video with 60s interval must produce exactly 10 images."""
        vfile = self._generate_video("test_03_10m.mp4", duration=600, fps=1)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 60,
            "extract_audio": True,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        # Timestamps: 0, 60, 120, 180, 240, 300, 360, 420, 480, 540 -> exactly 10 frames
        self.assertEqual(res["total_frames_extracted"], 10)

        out_path = Path(res["output_dir"])
        frames = sorted(list((out_path / "frames").glob("*.jpg")))
        self.assertEqual(len(frames), 10)

        expected_timestamps = [
            "00-00-00", "00-01-00", "00-02-00", "00-03-00", "00-04-00",
            "00-05-00", "00-06-00", "00-07-00", "00-08-00", "00-09-00",
        ]
        for i, expected_ts in enumerate(expected_timestamps, start=1):
            expected_name = f"frame_{i:06d}_{expected_ts}.jpg"
            self.assertEqual(frames[i - 1].name, expected_name)

        # Verify CSV contents
        csv_file = out_path / "metadata" / "frames.csv"
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 10)
            self.assertEqual(reader[0]["timestamp"], "00:00:00")
            self.assertEqual(reader[9]["timestamp"], "00:09:00")

    # -------------------------------------------------------------------------
    # Test 4: 1-hour video (3600 seconds)
    # -------------------------------------------------------------------------
    def test_04_one_hour_video(self):
        """Test processing of a 1-hour video with 600s interval."""
        vfile = self._generate_video("test_04_1h.mp4", duration=3600, fps=1)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 600,
            "extract_audio": False,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        # Timestamps: 0, 600, 1200, 1800, 2400, 3000 (6 frames)
        self.assertEqual(res["total_frames_extracted"], 6)

        out_path = Path(res["output_dir"])
        frames = sorted(list((out_path / "frames").glob("*.jpg")))
        self.assertEqual(len(frames), 6)
        self.assertTrue(frames[0].name.startswith("frame_000001_00-00-00"))
        self.assertTrue(frames[5].name.startswith("frame_000006_00-50-00"))

    # -------------------------------------------------------------------------
    # Test 5: Video without audio
    # -------------------------------------------------------------------------
    def test_05_video_without_audio(self):
        """Test that a video with no audio stream is processed without crashing."""
        vfile = self._generate_video("test_05_no_audio.mp4", duration=5, audio_streams=0)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "extract_audio": True,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["audio_status"], "SKIPPED_NO_AUDIO")
        out_path = Path(res["output_dir"])
        self.assertFalse((out_path / "audio" / "extracted_audio.wav").exists())
        self.assertTrue(res["total_frames_extracted"] > 0)

    # -------------------------------------------------------------------------
    # Test 6: Video containing speech (Vosk transcription test)
    # -------------------------------------------------------------------------
    def test_06_video_containing_speech(self):
        """Test speech recognition and transcript generation using local Vosk model."""
        speech_wav = project_root / "tests" / "test_speech.wav"
        vfile = self._generate_video("test_06_speech.mp4", duration=3, audio_wav=speech_wav)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 1,
            "extract_audio": True,
            "enable_transcription": True,
            "transcription_language": "en",
            "transcription_model_path": str(project_root / "models" / "vosk-en"),
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["audio_status"], "EXTRACTED")
        self.assertIn(res["transcription_status"], ["SUCCESS", "SUCCESS_NO_SPEECH"])

        out_path = Path(res["output_dir"])
        txt_path = out_path / "transcript" / "transcript.txt"
        json_path = out_path / "transcript" / "transcript.json"
        srt_path = out_path / "transcript" / "transcript.srt"

        self.assertTrue(txt_path.exists())
        self.assertTrue(json_path.exists())
        self.assertTrue(srt_path.exists())

        with open(json_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            self.assertIn("status", t_data)
            self.assertIn("segments", t_data)

    # -------------------------------------------------------------------------
    # Test 7: Video containing multiple audio streams
    # -------------------------------------------------------------------------
    def test_07_multiple_audio_streams(self):
        """Test selecting specific stream from video with multiple audio tracks."""
        vfile = self._generate_video("test_07_multi_audio.mp4", duration=5, audio_streams=2)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "extract_audio": True,
            "audio_stream_index": 1,  # Select second audio stream
            "preserve_original_audio": True,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["audio_status"], "EXTRACTED")

        out_path = Path(res["output_dir"])
        self.assertTrue((out_path / "audio" / "extracted_audio.wav").exists())
        # Check original audio was preserved
        orig_audio = list((out_path / "audio").glob("original_audio.*"))
        self.assertEqual(len(orig_audio), 1)

    # -------------------------------------------------------------------------
    # Test 8: Video with variable frame rate (VFR)
    # -------------------------------------------------------------------------
    def test_08_variable_frame_rate(self):
        """Test processing video with variable frame rate."""
        vfile = self._generate_video("test_08_vfr.mp4", duration=6, vfr=True)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["total_frames_extracted"] >= 3)

    # -------------------------------------------------------------------------
    # Test 9: Video containing non-English speech / alternate audio
    # -------------------------------------------------------------------------
    def test_09_non_english_or_alternate_audio(self):
        """Test audio extraction and graceful handling when language model is missing."""
        vfile = self._generate_video("test_09_alternate.mp4", duration=4, audio_frequency=500)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "extract_audio": True,
            "enable_transcription": True,
            "transcription_language": "hi",
            "transcription_model_path": "./models/vosk-hi-nonexistent",  # Model missing
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        # Video processing should still succeed even though transcription model was missing
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["audio_status"], "EXTRACTED")
        self.assertEqual(res["transcription_status"], "SKIPPED")

    # -------------------------------------------------------------------------
    # Test 10: Corrupted video file
    # -------------------------------------------------------------------------
    def test_10_corrupted_video_file(self):
        """Test robust error handling when presented with a corrupted or invalid file."""
        corrupt_file = self.test_dir / "test_10_corrupt.mp4"
        # Write junk bytes
        corrupt_file.write_bytes(b"JUNKDATA_NOT_A_VALID_MP4_VIDEO_HEADER_1234567890")

        config = {
            "output_directory": str(self.output_dir),
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(corrupt_file)

        # Should report FAILED gracefully without crashing the whole application
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("error", res)

        out_path = Path(res["output_dir"])
        metadata_file = out_path / "metadata" / "video_info.json"
        self.assertTrue(metadata_file.exists())
        with open(metadata_file, "r", encoding="utf-8") as f:
            m = json.load(f)
            self.assertEqual(m["processing_status"], "FAILED")

    # -------------------------------------------------------------------------
    # Test 11: Video with Unicode filename
    # -------------------------------------------------------------------------
    def test_11_unicode_filename(self):
        """Test video with spaces and non-ASCII Unicode characters in filename."""
        src = self._generate_video("test_11_5s.mp4", duration=5)
        unicode_file = self.test_dir / "vidéo_測試_🎬 test 11.mp4"
        shutil.copyfile(src, unicode_file)

        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(unicode_file)

        self.assertEqual(res["status"], "SUCCESS")
        # 5s video with 2s interval -> 0s, 2s, 4s (3 frames)
        self.assertEqual(res["total_frames_extracted"], 3)
        self.assertTrue(Path(res["output_dir"]).exists())

    # -------------------------------------------------------------------------
    # Test 12: Large / high-resolution video with resizing
    # -------------------------------------------------------------------------
    def test_12_high_resolution_resizing(self):
        """Test high-resolution video and aspect-ratio-preserving downscaling."""
        vfile = self._generate_video("test_12_1080p.mp4", duration=4, width=1920, height=1080)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "resize": "1280x720",  # Downscale to 720p
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        out_path = Path(res["output_dir"])
        frame1 = out_path / "frames" / "frame_000001_00-00-00.jpg"
        self.assertTrue(frame1.exists())

        # Inspect dimensions with ffprobe
        cmd = [
            self.ffprobe_bin, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-print_format", "json",
            str(frame1),
        ]
        out = json.loads(subprocess.check_output(cmd).decode("utf-8"))
        w = out["streams"][0]["width"]
        h = out["streams"][0]["height"]
        self.assertEqual(w, 1280)
        self.assertEqual(h, 720)

    # -------------------------------------------------------------------------
    # Test 13: Video shorter than requested extraction interval
    # -------------------------------------------------------------------------
    def test_13_video_shorter_than_interval(self):
        """Test that a video shorter than extraction interval extracts exactly 1 frame at 00:00:00."""
        vfile = self._generate_video("test_13_short.mp4", duration=5)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 60,  # Interval 60s > duration 5s
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(vfile)

        self.assertEqual(res["status"], "SUCCESS")
        # Exactly 1 frame at timestamp 0
        self.assertEqual(res["total_frames_extracted"], 1)
        out_path = Path(res["output_dir"])
        frames = list((out_path / "frames").glob("*.jpg"))
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0].name.startswith("frame_000001_00-00-00"))

    # -------------------------------------------------------------------------
    # Test 14: Repeated processing of the same file (collision avoidance & overwrite)
    # -------------------------------------------------------------------------
    def test_14_repeated_processing_collision_avoidance(self):
        """Verify that repeated processing creates unique numbered directories when overwrite=False."""
        vfile = self._generate_video("test_14_repeat.mp4", duration=4)
        config_no_overwrite = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "overwrite_existing": False,
        }
        processor = VideoProcessor(config_no_overwrite)

        res1 = processor.process(vfile)
        res2 = processor.process(vfile)

        self.assertEqual(res1["status"], "SUCCESS")
        self.assertEqual(res2["status"], "SUCCESS")

        dir1 = Path(res1["output_dir"])
        dir2 = Path(res2["output_dir"])

        # Directories must be distinct to avoid overwriting previous results
        self.assertNotEqual(dir1, dir2)
        self.assertTrue(dir1.exists())
        self.assertTrue(dir2.exists())
        self.assertTrue(dir1.name.startswith(vfile.stem))
        self.assertTrue(dir2.name.startswith(vfile.stem))


if __name__ == "__main__":
    unittest.main()
