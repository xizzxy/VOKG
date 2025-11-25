"""
Video processing utilities
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing video metadata
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("No video stream found")

        # Extract metadata
        format_info = data.get("format", {})
        metadata = {
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bit_rate": int(format_info.get("bit_rate", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
            "fps": eval_fps(video_stream.get("r_frame_rate", "0/1")),
            "total_frames": int(video_stream.get("nb_frames", 0)),
        }

        logger.debug(f"Extracted video metadata: {metadata}", video_path=str(video_path))
        return metadata

    except subprocess.CalledProcessError as e:
        logger.error(f"FFprobe failed: {e.stderr}", video_path=str(video_path))
        raise
    except Exception as e:
        logger.error(f"Failed to extract video metadata: {e}", video_path=str(video_path))
        raise


def eval_fps(fps_str: str) -> float:
    """
    Evaluate FPS from fraction string (e.g., "30000/1001")

    Args:
        fps_str: FPS as fraction string

    Returns:
        FPS as float
    """
    try:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    except:
        return 0.0


def validate_video_file(video_path: Path) -> tuple[bool, Optional[str]]:
    """
    Validate video file

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file exists
    if not video_path.exists():
        return False, "File does not exist"

    # Check file size
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum ({settings.MAX_VIDEO_SIZE_MB}MB)"

    # Check file extension
    ext = video_path.suffix.lower().lstrip(".")
    if ext not in settings.allowed_formats_list:
        return False, f"Invalid file format. Allowed: {', '.join(settings.allowed_formats_list)}"

    # Validate with ffprobe
    try:
        metadata = extract_video_metadata(video_path)
        if metadata["duration"] == 0:
            return False, "Invalid video: duration is 0"
        if metadata["width"] == 0 or metadata["height"] == 0:
            return False, "Invalid video: no valid dimensions"
        return True, None
    except Exception as e:
        return False, f"Invalid video file: {str(e)}"


def get_video_thumbnail(video_path: Path, output_path: Path, timestamp: float = 1.0) -> bool:
    """
    Extract thumbnail from video at specific timestamp

    Args:
        video_path: Path to video file
        output_path: Path to save thumbnail
        timestamp: Timestamp in seconds

    Returns:
        True if successful
    """
    try:
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        logger.debug(f"Generated thumbnail at {timestamp}s", video_path=str(video_path))
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate thumbnail: {e.stderr}")
        return False


def estimate_frame_count(video_path: Path, target_fps: Optional[float] = None) -> int:
    """
    Estimate number of frames to extract

    Args:
        video_path: Path to video file
        target_fps: Target FPS for extraction (None = keyframes only)

    Returns:
        Estimated frame count
    """
    try:
        metadata = extract_video_metadata(video_path)
        duration = metadata["duration"]
        original_fps = metadata["fps"]

        if target_fps is None:
            # Keyframes only - estimate ~1 keyframe per 2 seconds
            estimated = int(duration / 2)
        else:
            # Uniform sampling
            estimated = int(duration * target_fps)

        # Cap at maximum
        return min(estimated, settings.FRAME_EXTRACTION_MAX_FRAMES)

    except Exception as e:
        logger.error(f"Failed to estimate frame count: {e}")
        return settings.FRAME_EXTRACTION_MAX_FRAMES
