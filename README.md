# Offline Video Frame Extraction and Audio Transcription System

A production-ready, fully offline video processing system built with **Python 3.11+**, **FFmpeg**, **FFprobe**, **OpenCV**, and **Vosk**.

The application accepts virtually any video format, extracts images at configurable intervals or scene cuts, automatically creates structured per-video directories, retrieves technical metadata, converts audio to 16 kHz mono WAV, and transcribes spoken dialogue into formatted text, structured JSON, and SRT subtitles—**entirely locally, with zero internet connectivity and zero external cloud APIs**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Requirements](#system-requirements)
4. [Installation & Setup](#installation--setup)
   - [FFmpeg Installation (Linux, macOS, Windows)](#ffmpeg-installation)
   - [Virtual Environment & Dependencies](#virtual-environment--dependencies)
   - [Offline Vosk Speech Models](#offline-vosk-speech-models)
5. [Directory Architecture](#directory-architecture)
6. [Configuration (`config/settings.json`)](#configuration-configsettingsjson)
7. [Command-Line Interface (CLI) Usage](#command-line-interface-cli-usage)
   - [Mode A: Interval-Based Extraction (Default)](#mode-a-interval-based-extraction-default)
   - [Mode B: True Frame-by-Frame Extraction](#mode-b-true-frame-by-frame-extraction)
   - [Mode C: Custom Timestamps](#mode-c-custom-timestamps)
   - [Mode D: Scene Change Detection](#mode-d-scene-change-detection)
   - [Audio Extraction & Stream Selection](#audio-extraction--stream-selection)
   - [Offline Speech Transcription](#offline-speech-transcription)
   - [Batch & Recursive Processing](#batch--recursive-processing)
8. [Output Structure & Artifacts](#output-structure--artifacts)
   - [Frame Naming Convention](#frame-naming-convention)
   - [Frame Mapping (`frames.csv`)](#frame-mapping-framescsv)
   - [Metadata Inspection (`video_info.json`)](#metadata-inspection-video_infojson)
   - [Transcripts (`.txt`, `.json`, `.srt`)](#transcripts-txt-json-srt)
9. [Performance & Seeking Mechanics](#performance--seeking-mechanics)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [Known Limitations](#known-limitations)
12. [Automated Test Suite](#automated-test-suite)

---

## Project Overview

Modern cloud-based video and audio processing services expose sensitive data, require ongoing subscriptions, and cannot operate in air-gapped or confidential environments. This system provides an offline alternative that leverages FFmpeg's native decoding pipeline alongside local Python libraries:

- **100% Offline**: Operates without internet access, external APIs, cloud keys, or telemetry.
- **Resilient Batch Processing**: Automatically discovers videos, handles corrupt or damaged files without halting batch runs, and produces a tabular execution summary.
- **Deterministic Organization**: Dedicated output directories prevent name collisions and prevent silent data overwrites.

---

## Key Features

- **Wide Container Support**: Decodes `MP4`, `MKV`, `AVI`, `MOV`, `WEBM`, `FLV`, `WMV`, `MPEG/MPG`, `M4V`, `TS/MTS`, `3GP`, and any format supported by FFmpeg.
- **Multiple Extraction Modes**:
  - **Interval**: Fixed time step (e.g. every 60s, 10s, 1s, or fractional seconds).
  - **All-Frames**: Every decoded frame with memory-safe streaming and disk usage pre-checks.
  - **Custom Timestamps**: Explicit user-provided timestamps (`HH:MM:SS` or seconds).
  - **Scene Detection**: Algorithmic visual transition detection using FFmpeg scene filters.
- **Quality Preservation & Scaling**:
  - Outputs `JPEG` (configurable quality), `PNG` (lossless), or `WebP`.
  - Aspect-ratio preserving downscaling (`--resize 1920x1080`, `--resize 1280x720`) preventing unnecessary upscaling.
  - Pixel format normalization (`yuvj420p` fallback for wide codec compatibility).
- **Automated Image Validation**: Every frame is validated using OpenCV (`cv2.imread`) to verify decode integrity and non-zero dimensions before committing to the output ledger.
- **Audio Extraction**:
  - Automatically extracts audio to 16 kHz 16-bit mono PCM WAV for maximum speech-engine compatibility.
  - Optional preservation of original audio codec streams (`--preserve-original-audio`).
  - Multi-track stream selection (`--audio-stream-index`).
  - Graceful bypass when no audio track exists.
- **Offline Speech-to-Text**:
  - Integrated local Vosk engine.
  - Generates plain text with timestamps `[HH:MM:SS]`, structured JSON, and standard SubRip `.srt` subtitle files.
  - Streaming chunk-based recognition for arbitrary length files without RAM exhaustion.

---

## System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, Fedora, Arch), macOS 12+, or Windows 10/11.
- **Python**: Version 3.11 or higher.
- **FFmpeg & FFprobe**: Version 5.0+ (recommended 7.x or 8.x).
- **RAM**: Minimum 2 GB (4 GB+ recommended for batch or parallel processing).
- **Disk Space**: Depends on the extraction mode (e.g. interval extraction needs only ~5–20 MB per video; all-frames extraction requires adequate storage for thousands of individual images).

---

## Installation & Setup

### FFmpeg Installation

#### Ubuntu / Debian Linux:
```bash
sudo apt update
sudo apt install -y ffmpeg
```

#### macOS (Homebrew):
```bash
brew install ffmpeg
```

#### Windows:
Using Winget:
```powershell
winget install Gyan.FFmpeg
```
Or Chocolatey:
```powershell
choco install ffmpeg
```
*Ensure `ffmpeg` and `ffprobe` are added to your system `PATH`.*

*(Optional Fallback)*: If system FFmpeg is not installed, the repository can use `static-ffmpeg` via `pip install static-ffmpeg`.

---

### Virtual Environment & Dependencies

1. Clone or navigate to the repository directory:
```bash
cd /mnt/Development/Offline_Video_Frame
```

2. Create and activate a Python virtual environment:
```bash
python3 -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows (cmd):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

3. Install required Python packages:
```bash
pip install -r requirements.txt
```

The contents of `requirements.txt`:
```text
opencv-python-headless>=4.8.0
vosk>=0.3.45
numpy>=1.24.0
static-ffmpeg>=2.5
```

---

### Offline Vosk Speech Models

To enable speech-to-text, download any official offline Vosk acoustic model and extract it into the `models/` directory.

- **English Model** (Compact, ~40 MB):
  Extract to `models/vosk-en`
- **Hindi Model** (Compact, ~43 MB):
  Extract to `models/vosk-hi`
- **Urdu Model** (Compact, ~32 MB):
  Extract to `models/vosk-ur`

Model directory structure:
```text
models/
└── vosk-en/
    ├── am/
    ├── conf/
    ├── graph/
    └── ivector/
```

> **Note:** The application strictly looks for existing local directories and **never** attempts network downloads automatically.

---

## Directory Architecture

```text
Offline_Video_Frame/
│
├── config/
│   └── settings.json             # Global application configuration defaults
│
├── input/                        # Default input folder for input video files
│   └── sample_video.mp4
│
├── output/                       # Output root: dedicated folder per video
│   └── sample_video/
│       ├── frames/               # Extracted image frames
│       │   ├── frame_000001_00-00-00.jpg
│       │   ├── frame_000002_00-01-00.jpg
│       │   └── frame_000003_00-02-00.jpg
│       ├── audio/                # Extracted WAV & original audio
│       │   ├── extracted_audio.wav
│       │   └── original_audio.m4a (optional)
│       ├── transcript/           # Speech recognition artifacts
│       │   ├── transcript.txt
│       │   ├── transcript.json
│       │   └── transcript.srt
│       ├── metadata/             # Technical metadata & frame map
│       │   ├── video_info.json
│       │   └── frames.csv
│       └── logs/                 # Per-video processing execution log
│           └── processing.log
│
├── models/                       # Offline Vosk acoustic models
│   └── vosk-en/
│
├── scripts/                      # Core application source modules
│   ├── __init__.py
│   ├── main.py                   # CLI entrypoint & batch coordinator
│   ├── video_processor.py        # Single-video processing pipeline
│   ├── frame_extractor.py        # Frame extraction engine (all modes)
│   ├── audio_extractor.py        # Audio extraction and conversion
│   ├── transcription.py          # Offline Vosk speech-to-text
│   ├── metadata_extractor.py     # FFprobe inspection & validation
│   ├── file_manager.py           # Path handling, collisions, disk checks
│   └── utils.py                  # Helpers, timing, logging, exceptions
│
├── tests/                        # Comprehensive unit & integration tests
│   ├── test_system.py            # All 14 mandatory scenario test cases
│   └── test_modes.py             # Dedicated extraction mode tests
│
├── requirements.txt
└── README.md
```

---

## Configuration (`config/settings.json`)

Default parameters can be customized in `config/settings.json`:

```json
{
  "input_directory": "./input",
  "output_directory": "./output",
  "extraction_mode": "interval",
  "frame_interval_seconds": 60,
  "image_format": "jpg",
  "jpeg_quality": 95,
  "preserve_resolution": true,
  "extract_audio": true,
  "enable_transcription": false,
  "transcription_language": "en",
  "transcription_model_path": "./models/vosk-en",
  "save_metadata": true,
  "generate_csv": true,
  "generate_srt": true,
  "recursive_scan": false,
  "max_parallel_jobs": 2,
  "overwrite_existing": false
}
```

*Every setting in `settings.json` can be overridden at runtime via CLI arguments.*

---

## Command-Line Interface (CLI) Usage

### Mode A: Interval-Based Extraction (Default)

Extract one image every 60 seconds (default):
```bash
python scripts/main.py --input "input/sample_video.mp4" --interval 60
```

Extract frames every 10 seconds:
```bash
python scripts/main.py --input "input/sample_video.mp4" --interval 10
```

Extract frames every 1 second:
```bash
python scripts/main.py --input "input/sample_video.mp4" --interval 1
```

Extract frames with fractional sub-second intervals (e.g. every 0.5 seconds):
```bash
python scripts/main.py --input "input/sample_video.mp4" --interval 0.5
```

Extract with the explicit end-frame included:
```bash
python scripts/main.py --input "input/sample_video.mp4" --interval 60 --include-end-frame
```

> **Primary Expectation:** A 600-second (10-minute) video processed with `--interval 60` produces **exactly 10 timestamped frames**:
> `frame_000001_00-00-00.jpg` through `frame_000010_00-09-00.jpg`. No frame is created at 600s unless `--include-end-frame` is specified.

---

### Mode B: True Frame-by-Frame Extraction

Extracts every single decoded frame:
```bash
python scripts/main.py --input "input/sample_video.mp4" --mode all-frames
```

- Calculates estimated storage beforehand.
- Inspects available free disk space using `shutil.disk_usage`.
- Streams decoding directly through FFmpeg without loading uncompressed frames into RAM.

---

### Mode C: Custom Timestamps

Extract frames at specific, comma-separated timestamps:
```bash
python scripts/main.py --input "input/sample_video.mp4" --mode timestamps --timestamps "00:00:15,00:01:30,00:03:45,00:07:20"
```

You can also pass raw seconds:
```bash
python scripts/main.py --input "input/sample_video.mp4" --mode timestamps --timestamps "15,90,225,440"
```

Or pass a text file containing one timestamp per line:
```bash
python scripts/main.py --input "input/sample_video.mp4" --mode timestamps --timestamps "timestamps.txt"
```

---

### Mode D: Scene Change Detection

Extracts representative frames when visual scene transitions occur:
```bash
python scripts/main.py --input "input/sample_video.mp4" --mode scene --scene-threshold 0.3
```
- Sensitivity range: `0.0` (triggers on minor motion) to `1.0` (triggers only on complete visual changes). Default: `0.3`.
- Uses conventional FFmpeg scene filter heuristics (`select='gt(scene,THRESHOLD)'`).

---

### Audio Extraction & Stream Selection

Extract audio (enabled by default):
```bash
python scripts/main.py --input "input/sample_video.mp4" --extract-audio
```

Disable audio extraction:
```bash
python scripts/main.py --input "input/sample_video.mp4" --no-audio
```

Preserve an exact copy of the source audio stream in its original codec (e.g. AAC, MP3, Opus):
```bash
python scripts/main.py --input "input/sample_video.mp4" --preserve-original-audio
```

Select a specific audio track in multi-audio videos:
```bash
python scripts/main.py --input "input/sample_video.mp4" --audio-stream-index 1
```

---

### Offline Speech Transcription

Enable speech-to-text using local Vosk model:
```bash
python scripts/main.py --input "input/sample_video.mp4" --interval 60 --transcribe --language en --model-path "models/vosk-en"
```

Transcription outputs:
- `output/<video>/transcript/transcript.txt`
- `output/<video>/transcript/transcript.json`
- `output/<video>/transcript/transcript.srt`

---

### Batch & Recursive Processing

Process all videos in a folder:
```bash
python scripts/main.py --input "input/" --interval 60
```

Recursively scan nested subdirectories:
```bash
python scripts/main.py --input "input/" --recursive --interval 60
```

Run parallel batch jobs across multiple CPU workers:
```bash
python scripts/main.py --input "input/" --recursive --workers 4
```

Force overwrite existing directories rather than creating numbered variants:
```bash
python scripts/main.py --input "input/sample_video.mp4" --overwrite
```

Resize frames to 1080p or 720p while preserving aspect ratio:
```bash
python scripts/main.py --input "input/sample_video.mp4" --resize 1280x720
```

Select image format (`jpg`, `png`, `webp`) and quality:
```bash
python scripts/main.py --input "input/sample_video.mp4" --format webp --quality 90
```

---

## Output Structure & Artifacts

### Frame Naming Convention

Frames are saved using 6-digit zero-padded sequential indices and formatted timestamps:
```text
frame_000001_00-00-00.jpg
frame_000002_00-01-00.jpg
frame_000003_00-02-00.jpg
```
For sub-second intervals or frame-by-frame extraction, millisecond precision is automatically appended:
```text
frame_000001_00-00-00-000.jpg
frame_000002_00-00-00-500.jpg
frame_000003_00-00-01-000.jpg
```

---

### Frame Mapping (`frames.csv`)

Located at `output/<video>/metadata/frames.csv`:
```csv
frame_number,timestamp_seconds,timestamp,filename
1,0,00:00:00,frame_000001_00-00-00.jpg
2,60,00:01:00,frame_000002_00-01-00.jpg
3,120,00:02:00,frame_000003_00-02-00.jpg
```

---

### Metadata Inspection (`video_info.json`)

Located at `output/<video>/metadata/video_info.json`:
```json
{
  "original_filename": "sample_video.mp4",
  "file_path": "/mnt/Development/Offline_Video_Frame/input/sample_video.mp4",
  "file_size_bytes": 7409842,
  "file_size_human": "7.07 MB",
  "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
  "duration_seconds": 600.0,
  "duration_formatted": "00:10:00.000",
  "width": 320,
  "height": 240,
  "display_aspect_ratio": "4:3",
  "frame_rate": 25.0,
  "total_frame_count": 15000,
  "video_codec": "h264",
  "audio_codec": "aac",
  "audio_sample_rate": 44100,
  "audio_channels": 1,
  "audio_stream_count": 1,
  "audio_streams": [
    {
      "index": 1,
      "codec": "aac",
      "sample_rate": "44100",
      "channels": 1,
      "language": "und"
    }
  ],
  "creation_timestamp": "2026-11-16T14:00:00Z",
  "processing_timestamp": "2026-11-16T14:10:35.991447+00:00",
  "frame_extraction_interval": 60.0,
  "total_extracted_images": 10,
  "processing_duration_seconds": 1.23,
  "processing_status": "SUCCESS",
  "error_message": null
}
```

---

### Transcripts (`.txt`, `.json`, `.srt`)

#### Plain Text (`transcript.txt`):
```text
[00:00:05] Welcome to this video.

[00:00:12] Today we will explain the process.

[00:01:08] Let us start with the first step.
```

#### Subtitle File (`transcript.srt`):
```text
1
00:00:05,210 --> 00:00:08,450
Welcome to this video.

2
00:00:12,100 --> 00:00:15,300
Today we will explain the process.
```

#### Detailed JSON (`transcript.json`):
```json
{
  "audio_file": "extracted_audio.wav",
  "language": "en",
  "model_path": "/path/to/models/vosk-en",
  "segments_count": 2,
  "full_text": "Welcome to this video. Today we will explain the process.",
  "segments": [
    {
      "segment_index": 1,
      "start_seconds": 5.21,
      "end_seconds": 8.45,
      "start_formatted": "00:00:05",
      "end_formatted": "00:00:08",
      "text": "Welcome to this video",
      "words": [
        {"word": "welcome", "start": 5.21, "end": 5.82, "conf": 1.0},
        {"word": "to", "start": 5.85, "end": 6.01, "conf": 1.0},
        {"word": "this", "start": 6.03, "end": 6.31, "conf": 1.0},
        {"word": "video", "start": 6.35, "end": 6.95, "conf": 1.0}
      ]
    }
  ],
  "status": "SUCCESS"
}
```

---

## Performance & Seeking Mechanics

### Fast Seeking vs. Accurate Decoding

In FFmpeg, seeking behavior depends heavily on argument placement:

1. **Input Seeking (`-ss` before `-i`)**:
   - `ffmpeg -ss <time> -i <input> ...`
   - **How it works**: FFmpeg jumps to the keyframe (I-frame) closest to and preceding `<time>`, then decodes forward frame-by-frame until the exact requested timestamp is reached.
   - **Performance**: Near-instant seeking even in 4-hour 4K files.
   - **Accuracy**: Frame-accurate in modern FFmpeg versions (2.1+). This system uses this method by default.

2. **Output Seeking (`-ss` after `-i`)**:
   - `ffmpeg -i <input> -ss <time> ...`
   - **How it works**: FFmpeg decodes the entire video stream from frame 0 up to `<time>`.
   - **Performance**: Extremely slow on long files; high CPU overhead.
   - **Accuracy**: Identical to modern input seeking.

3. **EOF Boundary Protection**:
   Seeking exactly to the container's duration header (e.g. `t = 600.0s`) can trigger an EOF error because the final decoded frame presentation timestamp is typically `duration - (1 / fps)`. This system automatically clamps final-frame seek points to ensure the final frame decodes reliably without failure.

---

## Troubleshooting & FAQ

### 1. `FFmpegNotFoundError`
**Symptom**: `Required binary/binaries not found: ffmpeg, ffprobe.`
**Solution**: Ensure FFmpeg is installed and accessible in your shell. Run `ffmpeg -version` and `ffprobe -version`. If on Ubuntu, run `sudo apt install ffmpeg`. If in a virtual environment without root access, install `pip install static-ffmpeg`.

### 2. Audio extraction is skipped
**Symptom**: `audio_status: SKIPPED_NO_AUDIO`
**Explanation**: The video contains no audio stream (e.g. screen recording or silent surveillance footage). The system detects this via FFprobe and bypasses audio processing gracefully without crashing.

### 3. Speech transcription is skipped
**Symptom**: `transcription_status: SKIPPED`
**Explanation**: Either `--transcribe` was not enabled, or the local model path specified by `--model-path` was not found. Download an acoustic model into `models/vosk-en` and run with `--transcribe --model-path models/vosk-en`.

### 4. High disk space warning on `all-frames`
**Symptom**: `DiskSpaceError: Insufficient disk space for all-frames extraction`
**Explanation**: A 60 FPS 4K video can produce 216,000 frames per hour. The system prevents unexpected disk filling. Pass `--force` if you have verified that adequate storage is available.

---

## Known Limitations

1. **Speech Recognition Accuracy**: Offline Vosk models provide private, local transcription, but word error rates are affected by heavy background noise, multiple simultaneous speakers, or accents. Do not expect cloud-level transcription accuracy without tuning domain-specific acoustic and language models.
2. **Damaged Containers**: If a video file is severely truncated or lacks container index headers (e.g. an incomplete download without an `moov` atom), FFprobe will report `CorruptedVideoError`. The system logs the failure and continues processing the remaining batch queue.
3. **Variable Frame Rate (VFR) Timestamps**: For VFR content, frame intervals between presentation timestamps are not uniform. The system reports the nearest decodable presentation timestamp.

---

## Automated Test Suite

The system includes test coverage for all 14 mandatory operational conditions.

To execute the test suite:
```bash
./venv/bin/python -m unittest discover tests
```

### Verified Test Cases:
1. `test_01_ten_second_video`: 10-second video interval extraction.
2. `test_02_one_minute_video`: 1-minute video with 10s interval (6 frames).
3. `test_03_ten_minute_video_exact_ten_frames`: **600-second video produces exactly 10 images at 60s intervals (0s, 60s, 120s, 180s, 240s, 300s, 360s, 420s, 480s, 540s)**.
4. `test_04_one_hour_video`: 3600-second video extraction.
5. `test_05_video_without_audio`: Graceful bypass for silent videos.
6. `test_06_video_containing_speech`: Local speech recognition and SRT generation.
7. `test_07_multiple_audio_streams`: Target stream selection and original codec preservation.
8. `test_08_variable_frame_rate`: Robust handling of VFR streams.
9. `test_09_non_english_or_alternate_audio`: Missing model graceful handling.
10. `test_10_corrupted_video_file`: Rejection of corrupted files without crashing batch runs.
11. `test_11_unicode_filename`: Files with spaces and non-ASCII Unicode characters (`vidéo_測試_🎬 test 11.mp4`).
12. `test_12_high_resolution_resizing`: Downscaling to 720p while preserving aspect ratio.
13. `test_13_video_shorter_than_interval`: Video shorter than interval produces exactly 1 frame at 00:00:00.
14. `test_14_repeated_processing_collision_avoidance`: Non-destructive directory numbering when `overwrite=False`.
15. `test_mode_custom_timestamps`: Verification of Mode C.
16. `test_mode_all_frames`: Verification of Mode B.
17. `test_mode_scene_detection`: Verification of Mode D.
18. `test_interval_with_end_frame`: Boundary condition with `--include-end-frame`.
19. `test_webp_output_format`: WebP compression and decoding validation.

All 19 tests pass successfully.

