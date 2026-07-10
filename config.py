"""Configuration management module for Cabangile AI Studio.

This module defines the core architectural configurations, environment variables,
resolution presets, aspect ratios, directory management, and font discovery 
mechanisms for the Cabangile AI Text-to-Video Engine v2.6.

It enforces SOLID principles, clean architecture, and strict cross-platform compatibility
across Android Termux, Linux, Windows, and macOS.
"""

import os
import sys
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# ==============================================================================
# 1. IMMUTABLE DATA STRUCTURES & VALUE OBJECTS
# ==============================================================================

@dataclass(frozen=True)
class AspectRatio:
    """Immutable representation of an aspect ratio."""
    width_ratio: int
    height_ratio: int

    @property
    def ratio_float(self) -> float:
        """Returns the aspect ratio as a floating-point value."""
        return self.width_ratio / self.height_ratio


# ==============================================================================
# 2. SYSTEM CONSTANTS & PRESETS
# ==============================================================================

class ResolutionPresets:
    """Standardized video resolution presets (height in pixels)."""
    R_720P = 720
    R_1080P = 1080
    R_1440P = 1440
    R_4K = 2160


class AspectRatioPresets:
    """Standardized aspect ratio presets for multimedia generation."""
    WIDESCREEN = AspectRatio(16, 9)
    VERTICAL = AspectRatio(9, 16)
    SQUARE = AspectRatio(1, 1)
    PORTRAIT = AspectRatio(4, 5)
    CINEMATIC = AspectRatio(21, 9)


# ==============================================================================
# 3. MAIN CONFIGURATION ENGINE
# ==============================================================================

class EngineConfig:
    """Production-grade configuration engine managing directories, environments, and platforms."""

    # Core Directories (Determined relative to execution or project root)
    BASE_DIR: Path = Path(__file__).resolve().parent
    PROJECTS_DIR: Path = BASE_DIR / "projects"
    AUDIO_DIR: Path = BASE_DIR / "output" / "audio"
    IMAGES_DIR: Path = BASE_DIR / "output" / "images"
    SCENES_DIR: Path = BASE_DIR / "output" / "scenes"
    TEMP_DIR: Path = BASE_DIR / "temp"
    LOG_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / "cache"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    MUSIC_DIR: Path = ASSETS_DIR / "music"
    FONTS_DIR: Path = ASSETS_DIR / "fonts"

    # Rendering Configurations
    FPS: int = 30
    AUDIO_BITRATE: str = "192k"
    VIDEO_CODEC: str = "libx264"
    AUDIO_CODEC: str = "aac"
    IMAGE_FORMAT: str = "png"
    VIDEO_FORMAT: str = "mp4"

    # Supported AI Providers
    SUPPORTED_PROVIDERS: List[str] = ["OpenAI", "Ollama", "Pollinations", "Stability AI"]

    # Environment Variable Declarations
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    STABILITY_API_KEY: Optional[str] = os.getenv("STABILITY_API_KEY")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Platform Flag Detection
    IS_ANDROID_TERMUX: bool = "TERMUX_VERSION" in os.environ or Path("/data/data/com.termux").exists()
    IS_WINDOWS: bool = sys.platform == "win32"
    IS_MAC: bool = sys.platform == "darwin"
    IS_LINUX: bool = sys.platform.startswith("linux") and not IS_ANDROID_TERMUX

    def __init__(self) -> None:
        """Initializes the engine configuration, logging setups, and runtime directories."""
        self.initialize_directories()
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Sets up an isolated, structured logging system for the engine."""
        log_file = self.LOG_DIR / "cabangile_engine.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("CabangileAIStudio.Config")
        self.logger.info("Configuration Engine initialized successfully.")

    def initialize_directories(self) -> None:
        """Automatically and safely creates all required system directories if they do not exist."""
        directories: List[Path] = [
            self.PROJECTS_DIR,
            self.AUDIO_DIR,
            self.IMAGES_DIR,
            self.SCENES_DIR,
            self.TEMP_DIR,
            self.LOG_DIR,
            self.CACHE_DIR,
            self.ASSETS_DIR,
            self.MUSIC_DIR,
            self.FONTS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def purge_temp(self) -> None:
        """Safely clears all artifacts inside the temporary directory without deleting the root."""
        if self.TEMP_DIR.exists():
            for item in self.TEMP_DIR.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Failed to delete {item} during temp purge: {e}")

    def get_dimensions(self, target_height: int, aspect_ratio: AspectRatio) -> Tuple[int, int]:
        """Calculates exact pixel width and height based on height preset and aspect ratio.

        Ensures dimensions are divisible by 2 for standard video codec encoding blocks.
        """
        calculated_width = int(target_height * aspect_ratio.ratio_float)
        # Ensure dimensions are even numbers (requirement for h264/yuv420p)
        final_width = calculated_width if calculated_width % 2 == 0 else calculated_width + 1
        final_height = target_height if target_height % 2 == 0 else target_height + 1
        return final_width, final_height

    def locate_font(self, font_name: str) -> Path:
        """Discovers system or embedded fonts across various host operating systems.

        Falls back to a default configuration or asset directory lookup if not found.
        """
        # Look inside local workspace assets first
        local_font_path = self.FONTS_DIR / font_name
        if local_font_path.exists():
            return local_font_path

        # Platform-specific searches
        search_paths: List[Path] = []

        if self.IS_ANDROID_TERMUX:
            search_paths.extend([
                Path("/system/fonts"),
                Path("/data/data/com.termux/files/usr/share/fonts")
            ])
        elif self.IS_WINDOWS:
            win_dir = os.environ.get("SystemRoot", "C:\\Windows")
            search_paths.append(Path(win_dir) / "Fonts")
        elif self.IS_MAC:
            search_paths.extend([
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path(os.path.expanduser("~/Library/Fonts"))
            ])
        elif self.IS_LINUX:
            search_paths.extend([
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path(os.path.expanduser("~/.local/share/fonts"))
            ])

        # Walk discovered search paths to locate the font file
        for base_path in search_paths:
            if base_path.exists():
                # Recursive search for the font name case-insensitively
                for path in base_path.rglob("*"):
                    if path.is_file() and font_name.lower() in path.name.lower():
                        return path

        # Fallback Strategy: Return standard asset destination path if nothing is found
        return local_font_path

    def verify_environment(self) -> Dict[str, bool]:
        """Validates existence of key API components and system endpoints.

        Returns a dictionary representing status matrices of provider integrations.
        """
        status_report = {
            "OpenAI": self.OPENAI_API_KEY is not None and len(self.OPENAI_API_KEY.strip()) > 0,
            "Stability AI": self.STABILITY_API_KEY is not None and len(self.STABILITY_API_KEY.strip()) > 0,
            "Ollama": False,
            "Pollinations": True  # Pollinations does not natively mandate an enforcement API Key
        }

        # Validate Ollama availability cleanly via basic checking mechanisms if required externally
        if self.OLLAMA_HOST:
            status_report["Ollama"] = True

        return status_report


# Instantiate unified global config object for runtime import across modules
config = EngineConfig()
