"""Cabangile AI Text-to-Video Engine.

A production-quality engine that parses a script, generates AI-driven narration,
creates dynamic visual plates with hardcoded subtitles, compiles individual scene clips,
and builds a fully synchronized final MP4 video using FFmpeg.
"""

import datetime
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party libraries
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# 1. FIXED BOOTSTRAP ORDER: Ensure logging directory exists *before* initializing handlers
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/cabangile_engine.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("CabangileEngine")


@dataclass
class Scene:
    """Data model representing a single video scene."""
    index: int
    title: str
    narration: str
    duration: float = 0.0
    image_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None


class ProjectManager:
    """Handles directory provisioning, workspace isolation, and file cleanups."""

    def __init__(self, base_output_dir: str = "output") -> None:
        self.base_output_dir = Path(base_output_dir)
        self.assets_dir = Path("assets")
        self.temp_base_dir = Path("temp")
        self._ensure_base_directories()

    def _ensure_base_directories(self) -> None:
        for folder in [self.base_output_dir, self.assets_dir, self.temp_base_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def create_workspace(self) -> Path:
        workspace_id = f"job_{uuid.uuid4().hex}"
        workspace = self.temp_base_dir / workspace_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    @staticmethod
    def cleanup_workspace(workspace_path: Path) -> None:
        if workspace_path.exists() and workspace_path.is_dir():
            try:
                shutil.rmtree(workspace_path)
                logger.info(f"Successfully cleaned up workspace: {workspace_path}")
            except Exception as e:
                logger.error(f"Error cleaning up workspace {workspace_path}: {e}", exc_info=True)


class ScriptParser:
    """Parses text sequences separated by double returns into Scene structures."""

    @staticmethod
    def parse(text_script: str) -> List[Scene]:
        logger.info("Parsing script inputs...")
        scenes: List[Scene] = []
        raw_blocks = [block.strip() for block in text_script.strip().split("\n\n") if block.strip()]

        for idx, block in enumerate(raw_blocks, start=1):
            lines = block.split("\n")
            title = lines[0].replace("Title:", "").replace("##", "").strip()
            
            if len(lines) > 1:
                narration = " ".join([line.strip() for line in lines[1:]])
            else:
                narration = title
                title = f"Scene {idx}"
                
            scenes.append(Scene(index=idx, title=title, narration=narration))
        
        logger.info(f"Successfully parsed {len(scenes)} scenes.")
        return scenes


class AudioEngine:
    """Manages Text-To-Speech asset generation via gTTS."""

    def __init__(self, language: str = "en") -> None:
        self.language = language

    def generate_narration(self, scene: Scene, workspace: Path) -> Path:
        output_file = workspace / f"scene_{scene.index}_narration.mp3"
        logger.info(f"Generating narration audio for Scene {scene.index}...")
        
        try:
            tts = gTTS(text=scene.narration, lang=self.language, slow=False)
            tts.save(str(output_file))
        except Exception as e:
            logger.error(f"gTTS Generation failed. Check your internet connection: {e}")
            raise RuntimeError("Narration generation failed. Internet connection required for gTTS.")
            
        return output_file


class ImageEngine:
    """Generates 1920x1080 visual layouts with cross-platform font fallbacks."""

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)) -> None:
        self.width, self.height = resolution

    def _draw_dark_gradient(self) -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), "#111116")
        gradient_draw = ImageDraw.Draw(image)
        for y in range(self.height):
            factor = y / self.height
            r = int(16 + factor * 20)
            g = int(16 + factor * 24)
            b = int(24 + factor * 32)
            gradient_draw.line([(0, y), (self.width, y)], fill=(r, g, b))
        return image

    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0] if bbox else 0
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def _load_system_font(self, size: int) -> ImageFont.ImageFont:
        """Scans common system pathways across Windows, Mac, and Linux systems."""
        font_candidates = [
            "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", 
            "NotoSans-Regular.ttf", "Helvetica.ttf"
        ]
        for font_name in font_candidates:
            try:
                return ImageFont.truetype(font_name, size)
            except IOError:
                continue
        logger.warning("No system font candidates found. Dropping back to lower-quality default font layer.")
        return ImageFont.load_default()

    def create_scene_image(self, scene: Scene, workspace: Path) -> Path:
        logger.info(f"Generating image frame layout for Scene {scene.index}...")
        output_file = workspace / f"scene_{scene.index}_frame.png"
        
        img = self._draw_dark_gradient()
        draw = ImageDraw.Draw(img)
        
        margin = int(self.width * 0.1)
        max_content_width = self.width - (2 * margin)
        
        title_font = self._load_system_font(64)
        body_font = self._load_system_font(40)

        # Render Title Card Accentuation
        title_text = f"{scene.index}. {scene.title}"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_x = (self.width - title_w) // 2
        title_y = int(self.height * 0.25)
        draw.text((title_x, title_y), title_text, font=title_font, fill="#5c6bc0")

        # Hardcoded Subtitle Text Wrapping Layer Blocks
        wrapped_lines = self._wrap_text(scene.narration, body_font, max_content_width)
        
        current_y = int(self.height * 0.5)
        for line in wrapped_lines:
            line_bbox = draw.textbbox((0, 0), line, font=body_font)
            line_w = line_bbox[2] - line_bbox[0]
            line_h = line_bbox[3] - line_bbox[1]
            line_x = (self.width - line_w) // 2
            draw.text((line_x, current_y), line, font=body_font, fill="#E0E0E6")
            current_y += line_h + 20

        img.save(output_file)
        return output_file


class FFmpegEngine:
    """Interacts directly with underlying native system FFmpeg binaries."""

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self._verify_dependencies()

    def _verify_dependencies(self) -> None:
        """Asserts execution platform holds proper pipeline utilities."""
        for utility in ["ffmpeg", "ffprobe"]:
            if not shutil.which(utility):
                raise RuntimeError(f"Missing core dependency system module: '{utility}' must be on your system PATH.")

    def _execute_cmd(self, cmd: List[str]) -> str:
        logger.debug(f"Executing FFmpeg Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failure!\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            raise RuntimeError(f"FFmpeg error: {result.stderr.splitlines()[-1] if result.stderr else 'Unknown Exception'}")
        return result.stderr

    def get_audio_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nocorrect=1", str(audio_path)
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed parsing track audio duration metadata: {e}")
            return 5.0

    def render_scene_clip(self, scene: Scene, workspace: Path) -> Path:
        output_clip = workspace / f"scene_{scene.index}_output.mp4"
        duration_str = f"{scene.duration:.3f}"
        
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(scene.image_path),
            "-i", str(scene.audio_path),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            "-t", duration_str,
            str(output_clip)
        ]
        self._execute_cmd(cmd)
        return output_clip

    def concatenate_videos(self, clips: List[Path], output_path: Path) -> None:
        logger.info("Merging scene sequences into global output target...")
        manifest_path = output_path.parent / "manifest.txt"
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            for clip in clips:
                f.write(f"file '{clip.absolute().as_posix()}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(manifest_path),
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_path)
        ]
        try:
            self._execute_cmd(cmd)
        finally:
            if manifest_path.exists():
                os.remove(manifest_path)

    def mix_background_music(self, video_path: Path, music_path: Path, output_path: Path) -> None:
        logger.info("Injecting ambient soundscape backgrounds into master container...")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(music_path),
            "-filter_complex", "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first[a]",
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "[a]",
            "-movflags", "+faststart", str(output_path)
        ]
        self._execute_cmd(cmd)


class VideoEngine:
    """Facade coordinator orchestrating the full automation workflow pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.fps = self.config.get("fps", 30)
        self.lang = self.config.get("voice_language", "en")
        self.output_dir = Path(self.config.get("output_directory", "output"))
        
        self.pm = ProjectManager(base_output_dir=str(self.output_dir))
        self.audio_engine = AudioEngine(language=self.lang)
        self.image_engine = ImageEngine()
        self.ffmpeg_engine = FFmpegEngine(fps=self.fps)

    def generate(
        self, 
        text_script: str, 
        bg_music_path: Optional[str] = None, 
        voice_language: Optional[str] = None,
        fps: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Runs the video generation pipeline with dynamic configuration parameter support."""
        start_time = datetime.datetime.now()
        
        # Handle hot runtime updates if API variables are supplied
        if voice_language and voice_language != self.lang:
            self.lang = voice_language
            self.audio_engine = AudioEngine(language=self.lang)
            logger.info(f"Runtime parameter update: Voice language changed to '{self.lang}'")

        if fps and fps != self.fps:
            self.fps = fps
            self.ffmpeg_engine = FFmpegEngine(fps=self.fps)
            logger.info(f"Runtime parameter update: Performance framerate target configured to {self.fps} FPS")

        workspace = self.pm.create_workspace()
        
        def report(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        errors: List[str] = []
        final_video_name = f"render_{int(start_time.timestamp())}.mp4"
        final_output_path = self.output_dir / final_video_name
        total_duration = 0.0
        scenes: List[Scene] = []

        try:
            report("Parsing raw text timeline tracks...")
            scenes = ScriptParser.parse(text_script)
            if not scenes:
                raise ValueError("Script contains zero parseable structural token elements.")

            clip_paths: List[Path] = []

            for scene in scenes:
                report(f"Generating voice narrative track for scene {scene.index}...")
                scene.audio_path = self.audio_engine.generate_narration(scene, workspace)
                
                scene.duration = self.ffmpeg_engine.get_audio_duration(scene.audio_path)
                total_duration += scene.duration
                
                report(f"Composing overlay graphics text mapping layers for scene {scene.index}...")
                scene.image_path = self.image_engine.create_scene_image(scene, workspace)
                
                report(f"Rendering frame structures to clip storage blocks for scene {scene.index}...")
                scene.video_path = self.ffmpeg_engine.render_scene_clip(scene, workspace)
                clip_paths.append(scene.video_path)

            report("Combining independent video timelines together...")
            temp_combined_path = workspace / "raw_combined.mp4"
            self.ffmpeg_engine.concatenate_videos(clip_paths, temp_combined_path)

            if bg_music_path and Path(bg_music_path).exists():
                report("Mixing user-defined audio background stems...")
                self.ffmpeg_engine.mix_background_music(temp_combined_path, Path(bg_music_path), final_output_path)
            else:
                report("Exporting final clean video assets to output volumes...")
                shutil.copy(temp_combined_path, final_output_path)

            report("Processing complete.")
            success = True

        except Exception as e:
            error_msg = f"Pipeline execution failure: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            success = False
            final_output_path = Path("")

        finally:
            self.pm.cleanup_workspace(workspace)

        render_time = (datetime.datetime.now() - start_time).total_seconds()

        return {
            "success": success,
            "output_path": str(final_output_path.absolute()) if success else None,
            "duration": round(total_duration, 2),
            "total_scenes": len(scenes),
            "render_time": round(render_time, 2),
            "errors": errors
        }
