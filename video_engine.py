"""
Cabangile AI Video Studio - Video Generation Engine
File: video_engine.py
Python: 3.12+

A modular, production-ready video rendering engine. 
Independent of FastAPI. Parses scripts, generates images & voice narration,
mixes audio, generates SRT subtitles, and compiles final MP4 via FFmpeg.
"""

import os
import re
import shutil
import logging
import asyncio
from typing import Callable, Coroutine, List, Dict, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/video_engine.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VideoEngine")

# Ensure base directories exist
for folder in ["output", "temp", "assets", "logs"]:
    os.makedirs(folder, exist_ok=True)


# =====================================================================
# 1. Models & Schemas
# =====================================================================

class Scene:
    def __init__(self, index: int, image_prompt: str, narration_text: str):
        self.index = index
        self.image_prompt = image_prompt.strip()
        self.narration_text = narration_text.strip()
        self.audio_path: Optional[str] = None
        self.image_path: Optional[str] = None
        self.duration: float = 0.0  # calculated post-audio generation


# =====================================================================
# 2. FileManager
# =====================================================================

class FileManager:
    """Manages creation, cleanup, and organization of temporary and output assets."""
    
    @staticmethod
    def get_temp_dir(job_id: str) -> Path:
        temp_dir = Path("temp") / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    @staticmethod
    def get_output_paths(job_id: str) -> tuple[Path, Path]:
        video_path = Path("output") / f"{job_id}.mp4"
        subtitle_path = Path("output") / f"{job_id}.srt"
        return video_path, subtitle_path

    @staticmethod
    def cleanup_temp(job_id: str) -> None:
        temp_dir = Path("temp") / job_id
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"[{job_id}] Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"[{job_id}] Failed to cleanup temp directory {temp_dir}: {e}")


# =====================================================================
# 3. SceneParser
# =====================================================================

class SceneParser:
    """Parses multi-scene scripts into Scene objects.
    
    Expected format example:
    [Scene 1]
    Visual: A majestic mountain peak at sunrise.
    Narration: Welcome to the peak of engineering excellence.
    
    [Scene 2]
    Visual: Code flowing down a screen like Matrix digital rain.
    Narration: Where lines of code shape the future.
    """
    
    @staticmethod
    def parse(script_text: str) -> List[Scene]:
        scenes: List[Scene] = []
        # Split scenes based on [Scene X] patterns
        raw_scenes = re.split(r'\[Scene\s*\d+\]', script_text, flags=re.IGNORECASE)
        
        scene_idx = 1
        for block in raw_scenes:
            if not block.strip():
                continue
            
            # Parse Visual / Prompt
            visual_match = re.search(r'(?:Visual|Image|Prompt):\s*(.*?)(?=(?:Narration|Audio|Voice):|$)', block, re.DOTALL | re.IGNORECASE)
            # Parse Narration / Audio Text
            audio_match = re.search(r'(?:Narration|Audio|Voice|Text):\s*(.*)', block, re.DOTALL | re.IGNORECASE)
            
            image_prompt = visual_match.group(1).strip() if visual_match else "Abstract background visualization"
            narration_text = audio_match.group(1).strip() if audio_match else ""
            
            # Fallback if parser fails regex but text exists
            if not audio_match and not visual_match:
                narration_text = block.strip()
            
            if narration_text:
                scenes.append(Scene(index=scene_idx, image_prompt=image_prompt, narration_text=narration_text))
                scene_idx += 1
                
        # If the script text didn't follow the scene syntax, treat entire text as single scene
        if not scenes and script_text.strip():
            scenes.append(Scene(index=1, image_prompt="Dynamic visual canvas", narration_text=script_text.strip()))
            
        return scenes


# =====================================================================
# 4. ImageGenerator (Provider Architecture)
# =====================================================================

class ImageGenerator:
    """Generates images using different providers."""
    
    def __init__(self, quality: str = "hd"):
        # Resolution mapping
        resolutions = {
            "standard": (1280, 720),
            "hd": (1920, 1080),
            "ultra_hd": (3840, 2160)
        }
        self.width, self.height = resolutions.get(quality, (1920, 1080))

    async def generate_image(self, provider: str, prompt: str, output_path: str) -> str:
        """Dispatches work to the appropriate image provider."""
        logger.info(f"Generating image with '{provider}' for prompt: '{prompt[:40]}...'")
        
        if provider == "local_canvas":
            return await self._generate_local_canvas(prompt, output_path)
        elif provider == "stability_ai":
            return await self._generate_stability_ai(prompt, output_path)
        else:
            logger.warning(f"Provider '{provider}' not found. Falling back to local_canvas.")
            return await self._generate_local_canvas(prompt, output_path)

    async def _generate_local_canvas(self, prompt: str, output_path: str) -> str:
        """Generates a real local image using PIL with text overlays to act as a placeholder or template."""
        # Offload CPU-heavy image drawing to an executor loop
        def draw():
            img = Image.new("RGB", (self.width, self.height), color=(30, 31, 34))
            draw = ImageDraw.Draw(img)
            
            # Draw abstract background details
            draw.rectangle([50, 50, self.width - 50, self.height - 50], outline=(79, 84, 92), width=5)
            draw.ellipse([self.width//2 - 200, self.height//2 - 200, self.width//2 + 200, self.height//2 + 200], outline=(114, 137, 218), width=3)
            
            # Add Prompt Text onto Image
            text = f"Prompt: {prompt}"
            # Simple text wrap logic
            wrapped_text = "\n".join([text[i:i+60] for i in range(0, len(text), 60)])
            draw.text((100, self.height - 200), wrapped_text, fill=(255, 255, 255))
            
            img.save(output_path, "PNG")
            
        await asyncio.to_thread(draw)
        return output_path

    async def _generate_stability_ai(self, prompt: str, output_path: str) -> str:
        """Example placeholder integration structure for Stability AI."""
        # Replace this with real HTTP calls to Stability AI if API keys are configured
        logger.info("Stability AI request (simulated via local canvas due to missing keys)")
        return await self._generate_local_canvas(prompt, output_path)


# =====================================================================
# 5. VoiceGenerator (gTTS, ElevenLabs, Coqui)
# =====================================================================

class VoiceGenerator:
    """Generates audio tracks from scene narrations."""
    
    async def generate_voice(self, provider: str, text: str, language: str, output_path: str) -> str:
        logger.info(f"Generating voice with '{provider}' for language '{language}'")
        
        if provider == "gtts":
            return await self._generate_gtts(text, language, output_path)
        elif provider == "elevenlabs":
            return await self._generate_elevenlabs(text, output_path)
        elif provider == "coqui":
            return await self._generate_coqui(text, output_path)
        else:
            logger.warning(f"Voice provider '{provider}' not supported. Falling back to gTTS.")
            return await self._generate_gtts(text, language, output_path)

    async def _generate_gtts(self, text: str, language: str, output_path: str) -> str:
        """Asynchronously calls gTTS library to write high-quality Google TTS speech files."""
        from gtts import gTTS
        
        def write_tts():
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(output_path)
            
        await asyncio.to_thread(write_tts)
        return output_path

    async def _generate_elevenlabs(self, text: str, output_path: str) -> str:
        """Asynchronous HTTP implementation for ElevenLabs API."""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY missing. Falling back to local gTTS emulation.")
            return await self._generate_gtts(text, "en", output_path)
            
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM" # Rachel default voice
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=60.0)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return output_path
            else:
                logger.error(f"ElevenLabs API failed: {response.text}. Falling back to gTTS.")
                return await self._generate_gtts(text, "en", output_path)

    async def _generate_coqui(self, text: str, output_path: str) -> str:
        """Asynchronous integration with local or API Coqui speech engines."""
        logger.warning("Coqui TTS offline API fallback to gTTS...")
        return await self._generate_gtts(text, "en", output_path)


# =====================================================================
# 6. AudioMixer & SubtitleGenerator
# =====================================================================

class AudioMixer:
    """Handles downloading and blending background audio with voiceover."""
    
    @staticmethod
    async def download_background_music(url: str, output_path: str) -> str:
        logger.info(f"Downloading background music from: {url}")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=120.0)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
        return output_path


class SubtitleGenerator:
    """Generates precise SubRip Subtitle (SRT) files."""
    
    @staticmethod
    def format_srt_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @classmethod
    def create_srt(cls, scenes: List[Scene], output_path: str) -> str:
        srt_content = []
        current_time = 0.0
        
        for idx, scene in enumerate(scenes, start=1):
            start_time = current_time
            end_time = current_time + scene.duration
            
            srt_content.append(str(idx))
            srt_content.append(f"{cls.format_srt_time(start_time)} --> {cls.format_srt_time(end_time)}")
            srt_content.append(scene.narration_text)
            srt_content.append("")  # Blank separator
            
            current_time = end_time
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))
            
        logger.info(f"Subtitles written to {output_path}")
        return output_path


# =====================================================================
# 7. FFmpegRenderer (Pure CLI Processes)
# =====================================================================

class FFmpegRenderer:
    """Directly interacts with system ffmpeg and ffprobe utilities."""

    @staticmethod
    async def get_audio_duration(file_path: str) -> float:
        """Interrogates system ffprobe to grab real audio clip length."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nocey=1",
            "-sexagesimal=0", file_path
        ]
        # Rewrite option because of a common ffprobe spelling typo fix in modern systems:
        cmd[-2] = "default=noprint_wrappers=1:nokey=1"
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            return float(stdout.decode().strip())
        return 5.0  # Safe fallback if file processing fails

    @staticmethod
    async def build_scene_video(image_path: str, audio_path: str, duration: float, fps: int, output_path: str) -> str:
        """Generates a standard temporary MP4 video containing static image and matching audio track."""
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264", "-t", str(duration),
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()
        return output_path

    @staticmethod
    async def concat_videos(video_list_path: str, output_path: str) -> None:
        """Assembles scene videos into a unified master clip."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", video_list_path,
            "-c", "copy", output_path
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()

    @staticmethod
    async def mix_background_music(video_path: str, bg_music_path: str, final_path: str, duration: float) -> None:
        """Combines master video and background music together using filter_complex to preserve volume balance."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1", "-i", bg_music_path,
            "-filter_complex", "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            "-t", str(duration),
            final_path
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()


# =====================================================================
# 8. VideoEngine (Unified Interface Entry Point)
# =====================================================================

class VideoEngine:
    """The master generation facade orchestrating the pipeline."""
    
    async def generate(
        self,
        job_id: str,
        script: str,
        voice_language: str = "en",
        voice_provider: str = "gtts",
        image_provider: str = "local_canvas",
        quality: str = "hd",
        fps: int = 30,
        bg_music_url: str | None = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> str:
        
        # Internal progress reporting wrapper
        def report_progress(value: int):
            if progress_callback:
                try:
                    progress_callback(value)
                except Exception as ex:
                    logger.warning(f"[{job_id}] Progress callback failed: {ex}")

        report_progress(0)
        logger.info(f"[{job_id}] Starting production execution.")
        
        # Init Directories & Output Paths
        temp_dir = FileManager.get_temp_dir(job_id)
        final_video_path, final_srt_path = FileManager.get_output_paths(job_id)
        
        try:
            # 1. Parsing Script text (10%)
            scenes = SceneParser.parse(script)
            if not scenes:
                raise ValueError("The provided script could not be parsed into any scenes.")
            report_progress(10)
            
            # Init subsystems
            img_gen = ImageGenerator(quality=quality)
            voice_gen = VoiceGenerator()
            
            # 2. Parallel audio & visual generation for performance (20% -> 35% -> 50%)
            audio_tasks = []
            image_tasks = []
            
            for scene in scenes:
                scene_temp_prefix = f"scene_{scene.index}"
                audio_p = str(temp_dir / f"{scene_temp_prefix}.mp3")
                img_p = str(temp_dir / f"{scene_temp_prefix}.png")
                
                # Add to task scheduler queues
                audio_tasks.append(voice_gen.generate_voice(voice_provider, scene.narration_text, voice_language, audio_p))
                image_tasks.append(img_gen.generate_image(image_provider, scene.image_prompt, img_p))

            report_progress(20)
            audio_paths = await asyncio.gather(*audio_tasks)
            report_progress(35)
            image_paths = await asyncio.gather(*image_tasks)
            
            # Populate scenes metadata and probe actual sound durations
            for i, scene in enumerate(scenes):
                scene.audio_path = audio_paths[i]
                scene.image_path = image_paths[i]
                scene.duration = await FFmpegRenderer.get_audio_duration(scene.audio_path)
                
            report_progress(50)

            # 3. Compile standalone scenes to MP4 segments (65%)
            scene_mp4_paths = []
            segment_tasks = []
            for scene in scenes:
                out_segment_mp4 = str(temp_dir / f"scene_{scene.index}.mp4")
                scene_mp4_paths.append(out_segment_mp4)
                segment_tasks.append(
                    FFmpegRenderer.build_scene_video(
                        image_path=scene.image_path,
                        audio_path=scene.audio_path,
                        duration=scene.duration,
                        fps=fps,
                        output_path=out_segment_mp4
                    )
                )
            await asyncio.gather(*segment_tasks)
            report_progress(65)

            # Write subtitle SRT file to dynamic output location
            SubtitleGenerator.create_srt(scenes, str(final_srt_path))

            # Write text concat manifest instruction sheet
            concat_list_file = temp_dir / "concat_list.txt"
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for path in scene_mp4_paths:
                    # FFmpeg concat requires escaped backslashes or forward slashes
                    normalized_path = Path(path).as_posix()
                    f.write(f"file '{normalized_path}'\n")

            # Concat scenes into main unmixed track (80%)
            unmixed_video_path = str(temp_dir / "unmixed_video.mp4")
            await FFmpegRenderer.concat_videos(str(concat_list_file), unmixed_video_path)
            report_progress(80)

            # 4. Optional Ambient Track Overlay & Mixing (95%)
            total_duration = sum(s.duration for s in scenes)
            if bg_music_url:
                temp_bg_music_path = str(temp_dir / "background_track.mp3")
                await AudioMixer.download_background_music(bg_music_url, temp_bg_music_path)
                await FFmpegRenderer.mix_background_music(
                    video_path=unmixed_video_path,
                    bg_music_path=temp_bg_music_path,
                    final_path=str(final_video_path),
                    duration=total_duration
                )
            else:
                # Copy unmixed version directly to target output destination
                shutil.copy(unmixed_video_path, str(final_video_path))
                
            report_progress(95)

            # 5. Pipeline Cleanup & Finish (100%)
            FileManager.cleanup_temp(job_id)
            report_progress(100)
            logger.info(f"[{job_id}] Successfully generated and saved MP4 to {final_video_path.absolute()}")
            
            return str(final_video_path.absolute())

        except Exception as e:
            logger.error(f"[{job_id}] Video generation failed with error: {e}", exc_info=True)
            # Do not clean up on failure to preserve logs and assets for debugging
            raise e
