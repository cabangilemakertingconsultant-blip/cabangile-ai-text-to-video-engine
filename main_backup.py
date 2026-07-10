#!/usr/bin/env python3
"""
Cabangile AI Text-to-Video Engine (v2.6)
A production-grade, single-file video generation pipeline.
Fully optimized for Python 3.11+ and Termux on Android.
"""

import sys
import json
import time
import socket
import logging
import shutil
import subprocess
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import timedelta

# Third-party dependencies
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# =====================================================================
# CONFIGURATION & ENVIRONMENT MANAGEMENT
# =====================================================================

@dataclass(frozen=True)
class AspectRatio:
    width: int
    height: int
    name: str

class EngineConfig:
    """Manages system configurations, directory paths, and video presets."""
    BASE_DIR: Path = Path(__file__).resolve().parent / "output"
    AUDIO_DIR: Path = BASE_DIR / "audio"
    IMAGES_DIR: Path = BASE_DIR / "images"
    SCENES_DIR: Path = BASE_DIR / "scenes"
    TEMP_DIR: Path = BASE_DIR / "temp"
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "engine.log"

    # Base Aspect Ratio Templates
    ASPECT_RATIOS: Dict[str, Dict[str, int]] = {
        "16:9": {"w_ratio": 16, "h_ratio": 9, "name": "Landscape"},
        "9:16": {"w_ratio": 9, "h_ratio": 16, "name": "Portrait/Shorts"},
        "1:1": {"w_ratio": 1, "h_ratio": 1, "name": "Square"}
    }

    # Resolution Scaling Map
    QUALITY_PRESETS: Dict[str, int] = {
        "720p": 1280,
        "1080p": 1920,
        "4K": 3840
    }

    FPS: int = 24

    @classmethod
    def get_dimensions(cls, ratio_str: str, quality_str: str) -> AspectRatio:
        """Calculates precise dimensions based on ratio selection and vertical target bounds."""
        target_width = cls.QUALITY_PRESETS.get(quality_str, 1920)
        ratio_data = cls.ASPECT_RATIOS.get(ratio_str, cls.ASPECT_RATIOS["16:9"])
        
        # Calculate matching axis properties based on core rules
        w_r = ratio_data["w_ratio"]
        h_r = ratio_data["h_ratio"]
        
        if ratio_str == "16:9":
            width = target_width
            height = int((target_width / w_r) * h_r)
        elif ratio_str == "9:16":
            height = target_width
            width = int((target_width / h_r) * w_r)
        else: # 1:1
            width = target_width
            height = target_width
            
        # Ensure values remain cleanly divisible by 2 for standard macroblocks
        width = (width // 2) * 2
        height = (height // 2) * 2
        
        return AspectRatio(width, height, ratio_data["name"])

    @classmethod
    def initialize_directories(cls) -> None:
        """Builds clean pipeline directory trees."""
        dirs = [cls.BASE_DIR, cls.AUDIO_DIR, cls.IMAGES_DIR, cls.SCENES_DIR, cls.TEMP_DIR, cls.LOG_DIR]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

        # Configure logging engines to record structural events
        logging.basicConfig(
            filename=str(cls.LOG_FILE),
            level=logging.INFO,
            format="%(asctime)s - [%(levelname)s] - %(message)s",
            filemode="a"
        )

    @classmethod
    def purge_temporary_files(cls) -> None:
        """Cleans internal files inside temporary folders without breaking directories."""
        if cls.TEMP_DIR.exists():
            for asset in cls.TEMP_DIR.iterdir():
                try:
                    if asset.is_file() or asset.is_symlink():
                        asset.unlink()
                    elif asset.is_dir():
                        shutil.rmtree(asset)
                except Exception as e:
                    logging.warning(f"Failed to clear non-critical temporary asset {asset}: {e}")

# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class Scene:
    """Represents a structured cinematic video segment."""
    index: int
    title: str
    text: str
    prompt: str
    duration: float = 0.0
    audio_path: Optional[Path] = None
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None

# =====================================================================
# SUBPROCESS EXECUTION & VERIFICATION ENGINE
# =====================================================================

class EnvironmentVerifier:
    """Ensures host runtime environment complies with system constraints."""
    
    @staticmethod
    def run_safe_command(cmd: List[str], desc: str) -> subprocess.CompletedProcess:
        """Executes system commands while preserving detailed log trails for exceptions."""
        try:
            logging.info(f"Executing: {' '.join(cmd)}")
            with open(EngineConfig.LOG_FILE, "a") as log_output:
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=log_output, 
                    check=True, 
                    text=True
                )
            return result
        except (subprocess.SubprocessError, FileNotFoundError) as error:
            msg = f"Command failed during execution: {desc}. Error details: {error}"
            logging.error(msg)
            raise RuntimeError(msg) from error

    @classmethod
    def locate_binary(cls, binary_name: str) -> Path:
        """Locates active binary paths across standard paths and Termux-specific system folders."""
        system_path = shutil.which(binary_name)
        if system_path:
            return Path(system_path)
            
        # Termux environment path targets
        termux_path = Path(f"/data/data/com.termux/files/usr/bin/{binary_name}")
        if termux_path.exists():
            return termux_path
            
        raise FileNotFoundError(f"Required binary asset '{binary_name}' was not detected in local system environment chains.")

    @classmethod
    def verify_environment(cls) -> bool:
        """Validates systemic capability prerequisites for FFmpeg processing chains."""
        try:
            ffmpeg_path = cls.locate_binary("ffmpeg")
            ffprobe_path = cls.locate_binary("ffprobe")
            
            # Verify paths return basic telemetry cleanly
            subprocess.run([str(ffmpeg_path), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run([str(ffprobe_path), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception as env_error:
            logging.error(f"Environment baseline validation failed: {env_error}")
            return False

    @staticmethod
    def check_network(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
        """Verifies network layer interfaces are live for API lookups."""
        try:
            socket.setdefaulttimeout(timeout)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, port))
            return True
        except OSError:
            return False

# =====================================================================
# SCRIPT GENERATION LAYER
# =====================================================================

class StoryEngine:
    """Interfaces with script generation layers or local layout structures."""

    @staticmethod
    def generate_script(topic: str) -> List[Scene]:
        """Compiles clean automation timelines matching targeted topic definitions."""
        clean_topic = topic.strip().title()
        
        raw_scenes = [
            {
                "title": "Introduction",
                "text": f"Welcome to our operational breakdown of {clean_topic}. Today, we analyze the architectural foundations and system mechanics driving development.",
                "prompt": f"A minimalist high-tech workspace visualizing systemic integration of {clean_topic}, soft workspace key-lighting, depth of field, 4k resolution cinematic render"
            },
            {
                "title": "Core Mechanics",
                "text": "At the center of this paradigm lies an optimized automation logic layer. Eliminating processing latency requires continuous monitoring across data paths.",
                "prompt": "Abstract digital infrastructure nodes connecting dynamically under low key lighting, corporate blue and clean amber neon accents"
            },
            {
                "title": "Scalability Vectors",
                "text": f"When scaling out frameworks for {clean_topic}, engineering rules demand clean abstraction lines. High-throughput demands must never compromise storage integrity.",
                "prompt": "Symmetric metallic array systems stretching clean parallel visual lines into a soft dark horizon, micro-glow data relays"
            },
            {
                "title": "Future Optimization",
                "text": "Looking forward, the roadmap depends on decoupled deployment models. Designing adaptive logic engines ensures long term compatibility with evolving runtimes.",
                "prompt": "Clean futuristic data schematic overlaying a sharp dark structural backdrop, soft geometric presentation style"
            }
        ]

        # Dynamic local extraction fallbacks via local inference blocks
        try:
            url = "http://localhost:11434/api/chat"
            system_prompt = "You are a production assistant. Output a valid raw JSON list of objects matching schema keys: title, text, prompt."
            data = {
                "model": "llama3",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a short multi-scene production script about: {clean_topic}"}
                ],
                "stream": False,
                "format": "json"
            }
            req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=4) as response:
                res_data = json.loads(response.read().decode())
                content = json.loads(res_data['message']['content'])
                if isinstance(content, list) and len(content) > 0:
                    raw_scenes = content
        except Exception as e:
            logging.info(f"Ollama localized optimization bypass; operating engine via standard base blueprint layout. Details: {e}")

        return [
            Scene(index=i, title=data["title"], text=data["text"], prompt=data["prompt"])
            for i, data in enumerate(raw_scenes)
        ]

# =====================================================================
# AUDIO SYNTHESIS ENGINE
# =====================================================================

class NarrationEngine:
    """Manages text-to-speech rendering pipelines with integrated retry logic."""

    @staticmethod
    def render_speech(scene: Scene, output_dir: Path, max_retries: int = 3) -> Path:
        """Synthesizes voice narration tracks using progressive backoff routines to mitigate network drops."""
        target_path = output_dir / f"scene_{scene.index}_speech.mp3"
        
        for attempt in range(max_retries):
            try:
                tts = gTTS(text=scene.text, lang='en', tld='com', slow=False)
                tts.save(str(target_path))
                if target_path.exists() and target_path.stat().st_size > 0:
                    return target_path
            except Exception as e:
                logging.warning(f"TTS synthesis failure on scene {scene.index} (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logging.error(f"Critical execution error during voice synthesis for segment {scene.index}.")
                    raise e
        return target_path

# =====================================================================
# GRAPHICS ENGINE & TEXT WRAPPING
# =====================================================================

class ImageEngine:
    """Renders programmatic backdrops or text layouts optimized for mobile system memory constraints."""

    @staticmethod
    def _locate_system_font(size: int) -> ImageFont.ImageFont:
        """Finds clean TrueType fonts across known Android/Termux system storage paths."""
        font_targets = [
            "/system/fonts/Roboto-Regular.ttf",
            "/system/fonts/NotoSans-Regular.ttf",
            "/system/fonts/DroidSans.ttf",
            "/data/data/com.termux/files/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
        for font_path in font_targets:
            if Path(font_path).exists():
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text_to_bounds(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        """Wraps text strings dynamically based on font dimensions to prevent canvas clipping."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Handle text bounding checks cleanly across Pillow variations
            if hasattr(font, 'getbbox'):
                w = font.getbbox(test_line)[2]
            else:
                w = font.getsize(test_line)[0]
                
            if w <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    @classmethod
    def generate_canvas(cls, scene: Scene, aspect: AspectRatio, is_title: bool = False) -> Path:
        """Generates visual graphic layouts with responsive typography containment overlays."""
        width, height = aspect.width, aspect.height
        filename = f"scene_{scene.index}.png" if not is_title else "scene_title_card.png"
        target_path = EngineConfig.IMAGES_DIR / filename

        image_downloaded = False
        if not is_title:
            try:
                encoded_prompt = urllib.parse.quote(scene.prompt)
                api_url = f"https://image.pollinations.ai/p/{encoded_prompt}?width={width}&height={height}&model=flux&seed=42"
                req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=12) as response:
                    with open(target_path, "wb") as f:
                        f.write(response.read())
                image_downloaded = True
                logging.info(f"Successfully processed remote image asset pipeline for item: {filename}")
            except Exception as e:
                logging.warning(f"AI image download fallback triggered for scene {scene.index}: {e}")
                image_downloaded = False

        # Memory-safe image handling wrapper
        if not image_downloaded:
            # Procedural graphic engine generation pass
            with Image.new("RGBA", (width, height), color=(18, 24, 38, 255)) as img:
                draw = ImageDraw.Draw(img)
                # Render a subtle technical gradient overlay
                for y in range(0, height, 4):
                    intensity = int(25 + (y / height) * 25)
                    draw.line([(0, y), (width, y)], fill=(12, 18, intensity, 255), width=4)
                draw.rectangle([30, 30, width - 30, height - 30], outline=(255, 255, 255, 12), width=2)
                img.convert("RGB").save(target_path, "PNG")
                logging.info(f"Generated procedural graphics asset for fallbacks: {filename}")

        # Composing Typographic Treatments with Context Managers to save RAM
        with Image.open(target_path).convert("RGBA") as base_canvas:
            overlay = Image.new("RGBA", base_canvas.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            font_size = int(height * 0.045)
            font = cls._locate_system_font(font_size)
            
            if is_title:
                title_font = cls._locate_system_font(int(height * 0.065))
                text_to_render = scene.title.upper()
                margin = int(width * 0.1)
                wrapped_lines = cls._wrap_text_to_bounds(text_to_render, title_font, width - (margin * 2))
                
                # Render title card typography in the center of the frame
                total_h = len(wrapped_lines) * (font_size * 1.5)
                current_y = (height - total_h) / 2
                
                overlay_draw.rectangle([margin - 20, current_y - 20, width - margin + 20, current_y + total_h + 20], fill=(0, 0, 0, 180))
                for line in wrapped_lines:
                    overlay_draw.text((margin, current_y), line, font=title_font, fill=(255, 215, 0, 255))
                    current_y += int(font_size * 1.5)
            else:
                # Standard Scene Typography Layout Placement (Header Title Bar)
                text_to_render = scene.title.upper()
                overlay_draw.rectangle([40, 40, width - 40, int(height * 0.14)], fill=(0, 0, 0, 140))
                overlay_draw.text((60, 55), text_to_render, font=font, fill=(50, 210, 255, 255))

            with Image.alpha_composite(base_canvas, overlay) as final_composition:
                final_composition.convert("RGB").save(target_path, "PNG")
                
        return target_path

# =====================================================================
# MULTIMEDIA PIPELINE (FFMPEG LAYER)
# =====================================================================

class FFmpegEngine:
    """Manages programmatic interactions with local system FFmpeg binary installations."""

    @staticmethod
    def read_audio_duration(audio_path: Path) -> float:
        """Extracts exact metric runtime durations using ffprobe utility checks."""
        ffprobe_bin = EnvironmentVerifier.locate_binary("ffprobe")
        cmd = [
            str(ffprobe_bin), "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    @staticmethod
    def build_scene_clip(image_path: Path, audio_path: Path, duration: float, aspect: AspectRatio, output_path: Path) -> None:
        """Builds standardized independent mp4 scene segments optimized for mobile processors."""
        ffmpeg_bin = EnvironmentVerifier.locate_binary("ffmpeg")
        fade_duration = 0.4
        
        video_filter = (
            f"scale={aspect.width}:{aspect.height},"
            f"fade=t=in:st=0:d={fade_duration},"
            f"fade=t=out:st={duration - fade_duration}:d={fade_duration},"
            f"format=yuv420p"
        )
        
        # Optimized presets for mobile processors running under Termux environments
        cmd = [
            str(ffmpeg_bin), "-y", "-loop", "1", "-i", str(image_path), "-i", str(audio_path),
            "-vf", video_filter, "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p", "-t", f"{duration}", str(output_path)
        ]
        EnvironmentVerifier.run_safe_command(cmd, f"Synthesizing sequence module: {output_path.name}")

    @staticmethod
    def generate_procedural_background_music(duration: float, output_path: Path) -> None:
        """Generates localized dynamic synth background ambient padding streams."""
        ffmpeg_bin = EnvironmentVerifier.locate_binary("ffmpeg")
        audio_expression = "anoisesrc=color=pink:amplitude=0.008,lowpass=f=300[b];sine=frequency=60:sample_rate=44100[s];[b][s]amix=inputs=2:weights=1 0.2[out]"
        cmd = [
            str(ffmpeg_bin), "-y", "-f", "lavfi", "-i", audio_expression, "-t", f"{duration}", "-c:a", "mp3", str(output_path)
        ]
        EnvironmentVerifier.run_safe_command(cmd, "Generating low-volume ambient backing track patterns")

    @staticmethod
    def merge_all_scenes(scene_clips: List[Path], output_path: Path) -> None:
        """Stitches structural segment arrays together using absolute path assignments."""
        ffmpeg_bin = EnvironmentVerifier.locate_binary("ffmpeg")
        concat_file = EngineConfig.TEMP_DIR / "concat_manifest.txt"
        
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in scene_clips:
                f.write(f"file '{clip.resolve()}'\n")
                
        cmd = [
            str(ffmpeg_bin), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)
        ]
        EnvironmentVerifier.run_safe_command(cmd, "Stitching core production components into unified master format")

    @staticmethod
    def mix_background_with_ducking(video_input: Path, music_input: Path, output_path: Path) -> None:
        """Blends narration audio safely over background ambiance, with fallback support for basic setups."""
        ffmpeg_bin = EnvironmentVerifier.locate_binary("ffmpeg")
        complex_filter = "[1:a]volume=0.15[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        
        cmd = [
            str(ffmpeg_bin), "-y", "-i", str(video_input), "-i", str(music_input),
            "-filter_complex", complex_filter, "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(output_path)
        ]
        
        try:
            EnvironmentVerifier.run_safe_command(cmd, "Mixing audio channels with ducking filters")
        except RuntimeError as e:
            logging.warning(f"Standard audio amix failed or is unsupported. Falling back to direct audio stream retention. Details: {e}")
            # Safe recovery strategy: bypass background mix entirely and output the core project file directly
            shutil.copy(video_input, output_path)

    @staticmethod
    def burn_subtitles_to_stream(video_input: Path, srt_path: Path, output_path: Path) -> None:
        """Hardburns target production subtitles using clean absolute path formats."""
        ffmpeg_bin = EnvironmentVerifier.locate_binary("ffmpeg")
        
        # Absolute path clean sanitization mappings tailored specifically for standard filter strings
        escaped_srt = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")
        
        cmd = [
            str(ffmpeg_bin), "-y", "-i", str(video_input),
            "-vf", f"subtitles='{escaped_srt}':force_style='Alignment=2,FontSize=16,OutlineColour=&Haa000000,BorderStyle=3,MarginV=25'",
            "-c:a", "copy", str(output_path)
        ]
        try:
            EnvironmentVerifier.run_safe_command(cmd, "Burning generated subtitles into video track output frames")
        except RuntimeError as filter_err:
            logging.warning(f"Subtitle filtering bypass triggered. Android build might lack libass. Exporting raw video stream. Details: {filter_err}")
            shutil.copy(video_input, output_path)

# =====================================================================
# SUBTITLE GENERATION UTILITIES
# =====================================================================

class SubtitleGenerator:
    """Assembles chronological text tracks mapping directly to video runtime markers."""

    @staticmethod
    def format_srt_timestamp(seconds: float) -> str:
        """Converts numeric seconds into conforming SRT format structures."""
        td = timedelta(seconds=seconds)
        total_secs = int(td.total_seconds())
        hours = total_secs // 3600
        minutes = (total_secs % 3600) // 60
        secs = total_secs % 60
        millis = int((seconds - total_secs) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @classmethod
    def generate_srt(cls, scenes: List[Scene], title_duration: float) -> Path:
        """Builds and records external subtitle SRT timeline mapping files."""
        srt_path = EngineConfig.BASE_DIR / "final_subtitles.srt"
        current_timeline = title_duration
        
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, scene in enumerate(scenes, start=1):
                start_str = cls.format_srt_timestamp(current_timeline)
                current_timeline += scene.duration
                end_str = cls.format_srt_timestamp(current_timeline)
                
                f.write(f"{i}\n{start_str} --> {end_str}\n{scene.text}\n\n")
        return srt_path

# =====================================================================
# SYSTEM CORE CONVERGENCE PIPELINE MANAGER
# =====================================================================

class VideoEngine:
    """Orchestrates asset compilation stages from source data down to delivery formats."""

    def __init__(self) -> None:
        self.current_step = 1
        self.total_steps = 10  # Initial baseline steps, updated dynamically once scenes are built

    def progress_update(self, stage_description: str) -> None:
        """Outputs explicit status and completion telemetry updates to the console."""
        percentage = int((self.current_step / self.total_steps) * 100)
        print(f"Progress: [{percentage}%] | Stage {self.current_step}/{self.total_steps}: {stage_description}")
        sys.stdout.flush()
        self.current_step += 1

    def execute_pipeline(self) -> None:
        """Orchestrates structural asset generation pipelines from source down to delivery formats."""
        print("====================================================")
        print("      CABANGILE AI TEXT-TO-VIDEO ENGINE v2.6        ")
        print("          Engine Environment: Android/Termux        ")
        print("====================================================\n")

        # Initialize core environment validation routines
        EngineConfig.initialize_directories()
        
        if not EnvironmentVerifier.verify_environment():
            print("[CRITICAL ERROR] Host system missing functional 'ffmpeg' or 'ffprobe' builds inside execution chains.")
            print("Please run: pkg install ffmpeg inside your Termux window before executing this module.")
            sys.exit(1)

        topic = input("Enter video topic/concept: ").strip()
        if not topic:
            topic = "Automated Systems Engineering"

        print("\nSelect Aspect Ratio Preset Option:")
        print("1. 16:9 (Cinematic Landscape)")
        print("2. 9:16 (Vertical Shorts/Reels/TikTok)")
        print("3. 1:1  (Square Platform Feeds)")
        ratio_choice = input("Enter selection index (1-3) [Default: 1]: ").strip()
        
        ratio_mapping = {"2": "9:16", "3": "1:1"}
        selected_ratio = ratio_mapping.get(ratio_choice, "16:9")

        print("\nSelect Target Resolution Quality Preset:")
        print("1. 720p  (Highly optimized for mobile memory profiles)")
        print("2. 1080p (Standard High Definition Output)")
        print("3. 4K    (High Performance Render Profile)")
        quality_choice = input("Enter selection index (1-3) [Default: 2]: ").strip()
        
        quality_mapping = {"1": "720p", "3": "4K"}
        selected_quality = quality_mapping.get(quality_choice, "1080p")

        # Determine target aspect frame sizes
        aspect = EngineConfig.get_dimensions(selected_ratio, selected_quality)
        logging.info(f"Target rendering bounds initialized to configuration targets: {aspect.width}x{aspect.height} ({aspect.name})")

        # Dynamically evaluate script generation chains
        self.progress_update("Querying AI engine layout layers and setting up timelines...")
        scenes = StoryEngine.generate_script(topic)
        
        # Calculate strict progressive task limits based on real scene array allocations
        # Calculation: 3 initialization steps + (scenes * 4 processing subtasks) + 4 final assembly steps
        self.total_steps = 3 + (len(scenes) * 4) + 4

        # Clear active temporary layout frames safely
        EngineConfig.purge_temporary_files()

        # Step 2: Establish custom production Title Cards
        self.progress_update("Generating programmatic cinematic Title Card elements...")
        title_card_scene = Scene(index=99, title=topic, text="", prompt="")
        title_image = ImageEngine.generate_canvas(title_card_scene, aspect, is_title=True)
        
        title_audio_track = EngineConfig.TEMP_DIR / "title_silence.mp3"
        ffmpeg_bin = EnvironmentVerifier.locate_binary("ffmpeg")
        
        # Create silent spacing frame block segment
        subprocess.run(
            [str(ffmpeg_bin), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "3.0", "-c:a", "mp3", str(title_audio_track)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        
        title_video_clip = EngineConfig.TEMP_DIR / "title_segment.mp4"
        FFmpegEngine.build_scene_clip(title_image, title_audio_track, 3.0, aspect, title_video_clip)

        compiled_clips: List[Path] = [title_video_clip]
        accumulated_duration = 3.0

        # Step 3: Run segment synthesis routines
        for scene in scenes:
            self.progress_update(f"[Scene {scene.index + 1}] Processing text-to-speech audio tracking maps...")
            audio_path = NarrationEngine.render_speech(scene, EngineConfig.AUDIO_DIR)
            scene.audio_path = audio_path

            self.progress_update(f"[Scene {scene.index + 1}] Reading structural timeline block limits...")
            duration = FFmpegEngine.read_audio_duration(audio_path)
            scene.duration = duration
            accumulated_duration += duration

            self.progress_update(f"[Scene {scene.index + 1}] Running graphic layout processing passes...")
            image_path = ImageEngine.generate_canvas(scene, aspect)
            scene.image_path = image_path

            self.progress_update(f"[Scene {scene.index + 1}] Rendering independent video segment assets...")
            scene_video_output = EngineConfig.SCENES_DIR / f"scene_{scene.index}.mp4"
            FFmpegEngine.build_scene_clip(image_path, audio_path, duration, aspect, scene_video_output)
            scene.video_path = scene_video_output
            compiled_clips.append(scene_video_output)

        # Final System Assembly Sequences
        self.progress_update("Combining independent production clips into a raw video master...")
        raw_master_file = EngineConfig.TEMP_DIR / "raw_master_output.mp4"
        FFmpegEngine.merge_all_scenes(compiled_clips, raw_master_file)

        self.progress_update("Synthesizing dynamic low-volume ambient soundtrack layers...")
        ambient_audio_track = EngineConfig.TEMP_DIR / "ambient_soundtrack.mp3"
        FFmpegEngine.generate_procedural_background_music(accumulated_duration, ambient_audio_track)

        self.progress_update("Merging video master audio tracks with ambient backing fields...")
        mixed_master_file = EngineConfig.TEMP_DIR / "mixed_master_output.mp4"
        FFmpegEngine.mix_background_with_ducking(raw_master_file, ambient_audio_track, mixed_master_file)

        self.progress_update("Compiling timed SRT standard subtitle structures...")
        srt_file = SubtitleGenerator.generate_srt(scenes, title_duration=3.0)

        self.progress_update("Burning production subtitles onto output video frames...")
        final_delivery_video = EngineConfig.BASE_DIR / "final_production_video.mp4"
        FFmpegEngine.burn_subtitles_to_stream(mixed_master_file, srt_file, final_delivery_video)

        # Final Cleanup Sequence
        logging.info("Pipeline processing complete. Commencing active cleanup passes.")
        EngineConfig.purge_temporary_files()

        print("\n====================================================")
        print("PRODUCER EXECUTION SUCCESSFUL!")
        print(f"Output Master Location: {final_delivery_video.resolve()}")
        print(f"System Log Audit File:  {EngineConfig.LOG_FILE.resolve()}")
        print("====================================================")

# =====================================================================
# GLOBAL COUPLING APPLICATION RUNTIME INTERFACES
# =====================================================================

if __name__ == "__main__":
    try:
        pipeline_orchestrator = VideoEngine()
        pipeline_orchestrator.execute_pipeline()
    except KeyboardInterrupt:
        print("\n[ABORTED] Production pipeline cancelled by user request.")
        sys.exit(130)
    except Exception as fatal_exception:
        logging.critical("Pipeline thread crash recorded by global runtime handler.", exc_info=True)
        print(f"\n[FATAL SYSTEM EXCEPTION CAUGHT]: {fatal_exception}")
        print(f"Review full error traces inside execution log file: {EngineConfig.LOG_FILE}")
        sys.exit(1)
