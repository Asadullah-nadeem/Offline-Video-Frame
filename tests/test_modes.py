"""Unit and integration tests for extraction modes: Interval, All-Frames, Timestamps, Scene."""

import csv
import json
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
from scripts.utils import verify_ffmpeg


class TestExtractionModes(unittest.TestCase):
    """Test all frame extraction modes and formatting options."""

    @classmethod
    def setUpClass(cls):
        cls.ffmpeg_bin, cls.ffprobe_bin = verify_ffmpeg()
        cls.test_dir = project_root / "tests" / "test_media"
        cls.output_dir = project_root / "tests" / "test_outputs"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate a 6-second video with 2 scenes (3s blue, 3s red)
        cls.scene_video = cls.test_dir / "test_mode_scene.mp4"
        if not cls.scene_video.exists():
            cmd = [
                cls.ffmpeg_bin, "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=3",
                "-f", "lavfi", "-i", "color=c=red:s=320x240:d=3",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
                "-map", "[v]",
                "-c:v", "libx264", "-preset", "ultrafast",
                str(cls.scene_video),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Generate a 4-second standard video
        cls.std_video = cls.test_dir / "test_mode_std.mp4"
        if not cls.std_video.exists():
            cmd = [
                cls.ffmpeg_bin, "-y",
                "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=10",
                "-f", "lavfi", "-i", "sine=frequency=1000:duration=4",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                str(cls.std_video),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    def test_mode_custom_timestamps(self):
        """Mode C: Test explicit custom timestamps extraction."""
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "timestamps",
            "timestamps": "00:00:01,00:00:02.500,00:00:03",
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(self.std_video)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_frames_extracted"], 3)

        out_path = Path(res["output_dir"])
        frames = sorted(list((out_path / "frames").glob("*.jpg")))
        self.assertEqual(len(frames), 3)

        csv_path = out_path / "metadata" / "frames.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["timestamp"], "00:00:01.000")
            self.assertEqual(rows[1]["timestamp"], "00:00:02.500")
            self.assertEqual(rows[2]["timestamp"], "00:00:03.000")

    def test_mode_all_frames(self):
        """Mode B: Test true frame-by-frame extraction."""
        # 4s video at 10 fps contains 40 frames
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "all-frames",
            "image_format": "png",
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(self.std_video)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_frames_extracted"], 40)

        out_path = Path(res["output_dir"])
        frames = list((out_path / "frames").glob("*.png"))
        self.assertEqual(len(frames), 40)

    def test_mode_scene_detection(self):
        """Mode D: Test scene change detection."""
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "scene",
            "scene_threshold": 0.3,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(self.scene_video)

        self.assertEqual(res["status"], "SUCCESS")
        # Should detect the transition at 3.0s
        self.assertTrue(res["total_frames_extracted"] >= 1)

    def test_interval_with_end_frame(self):
        """Mode A: Test include_end_frame flag."""
        # 4s video with 1.5s interval without end frame -> 0s, 1.5s, 3.0s (3 frames)
        # With include_end_frame -> 0s, 1.5s, 3.0s, 4.0s (4 frames)
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 1.5,
            "include_end_frame": True,
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(self.std_video)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_frames_extracted"], 4)

        out_path = Path(res["output_dir"])
        frames = sorted(list((out_path / "frames").glob("*.jpg")))
        self.assertEqual(len(frames), 4)

    def test_webp_output_format(self):
        """Test WebP image output format."""
        config = {
            "output_directory": str(self.output_dir),
            "extraction_mode": "interval",
            "frame_interval_seconds": 2,
            "image_format": "webp",
            "overwrite_existing": True,
        }
        processor = VideoProcessor(config)
        res = processor.process(self.std_video)

        self.assertEqual(res["status"], "SUCCESS")
        out_path = Path(res["output_dir"])
        webp_frames = list((out_path / "frames").glob("*.webp"))
        self.assertTrue(len(webp_frames) > 0)


if __name__ == "__main__":
    unittest.main()

