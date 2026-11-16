"""Directory and file management for offline video processing."""

import logging
import re
import shutil
from pathlib import Path
from typing import List, Tuple

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
    ".mpeg", ".mpg", ".m4v", ".ts", ".mts", ".3gp", ".m2ts",
    ".vob", ".ogv"
}


def is_video_file(path: Path) -> bool:
    """Check if the given path has a supported video extension."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def find_video_files(input_path: Path | str, recursive: bool = False) -> List[Path]:
    """Discover video files from a single file path or directory.
    
    Args:
        input_path: Path to a video file or directory of videos.
        recursive: Whether to scan subdirectories recursively.
        
    Returns:
        List of Path objects for discovered video files, sorted.
        
    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If input is a directory and no video files are found.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        # Even if the extension is unusual, if specified directly by the user, we accept it
        return [path]

    if path.is_dir():
        files: List[Path] = []
        pattern = "**/*" if recursive else "*"
        for item in path.glob(pattern):
            if item.is_file() and not item.name.startswith("."):
                if item.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    files.append(item)

        files.sort(key=lambda p: str(p).lower())
        return files

    raise ValueError(f"Input path is neither a regular file nor a directory: {path}")


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for folder creation, preserving Unicode and spaces safely."""
    # Replace dangerous path characters with underscore
    sanitized = re.sub(r'[\/\\:\*\?"<>\|]', "_", filename)
    # Strip leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    return sanitized if sanitized else "unnamed_video"


def create_output_structure(
    base_output_dir: Path | str,
    video_path: Path,
    overwrite: bool = False,
) -> Tuple[Path, dict[str, Path]]:
    """Create an organized output directory structure for a video.
    
    Structure:
      <base_output_dir>/<video_stem>/
        ├── frames/
        ├── audio/
        ├── transcript/
        ├── metadata/
        └── logs/

    If overwrite is False and folder exists, appends _1, _2, etc. to prevent collisions.
    
    Args:
        base_output_dir: Base directory for output (e.g., ./output).
        video_path: Path to the video file being processed.
        overwrite: If True, reuse existing folder instead of creating a numbered variant.
        
    Returns:
        Tuple of (video_output_dir, dict of subfolder paths).
    """
    base_dir = Path(base_output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    folder_name = sanitize_filename(video_path.stem)
    video_output_dir = base_dir / folder_name

    if not overwrite:
        counter = 1
        while video_output_dir.exists():
            video_output_dir = base_dir / f"{folder_name}_{counter}"
            counter += 1

    subdirs = {
        "frames": video_output_dir / "frames",
        "audio": video_output_dir / "audio",
        "transcript": video_output_dir / "transcript",
        "metadata": video_output_dir / "metadata",
        "logs": video_output_dir / "logs",
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for item in subdir.iterdir():
                if item.is_file():
                    try:
                        item.unlink()
                    except Exception:
                        pass

    return video_output_dir, subdirs


def check_disk_space(target_path: Path | str, required_bytes: int) -> Tuple[bool, int, int]:
    """Check if the disk holding target_path has at least required_bytes free.
    
    Returns:
        (has_enough_space, free_bytes, required_bytes)
    """
    path = Path(target_path).resolve()
    check_dir = path if path.is_dir() else path.parent
    if not check_dir.exists():
        check_dir = check_dir.parent

    try:
        usage = shutil.disk_usage(check_dir)
        free_bytes = usage.free
        return (free_bytes >= required_bytes), free_bytes, required_bytes
    except Exception:
        # If disk usage check fails (e.g. exotic mount), assume space is available
        return True, 0, required_bytes


def cleanup_directory(directory: Path | str, ignore_errors: bool = True) -> None:
    """Safely remove a directory and all its contents."""
    path = Path(directory)
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=ignore_errors)

