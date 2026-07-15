# video_engine.py
import os
import subprocess
import tempfile
import logging
from typing import Optional, Callable
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import cloudinary
import cloudinary.uploader

logger = logging.getLogger("cabangile_api.video_engine")

# Configure Cloudinary using environment variables
# Ensure CLOUDINARY_URL is set in your system environment.
cloudinary.config(secure=True)

class VideoEngine:
    def assemble_video(
        self, 
        job_id: str, 
        script_text: str, 
        lang: str, 
        fps: int, 
        width: int, 
        height: int, 
        bitrate: str, 
        music_file: Optional[str],
        progress_callback: Callable[[int, str], None]
    ) -> str:
        """
        Assembles video assets using gTTS, Pillow, and FFmpeg, uploads the output
        to Cloudinary, and returns the secure hosted URL.
        """
        # Create a temporary working directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"[{job_id}] Initializing temporary workspace at {temp_dir}")
            
            # --- Step 1: Voiceover Generation ---
            progress_callback(15, "Generating voiceover narration via gTTS...")
            audio_path = os.path.join(temp_dir, "voice.mp3")
            try:
                tts = gTTS(text=script_text, lang=lang)
                tts.save(audio_path)
            except Exception as e:
                logger.error(f"[{job_id}] gTTS generation failed: {e}")
                raise RuntimeError(f"Voiceover generation failed: {e}")

            # Determine audio length to set video duration
            duration = self._get_audio_duration(audio_path)
            logger.info(f"[{job_id}] Generated audio duration: {duration:.2f} seconds")

            # --- Step 2: Visual Canvas Generation ---
            progress_callback(40, "Generating visual canvas frames...")
            image_path = os.path.join(temp_dir, "canvas.png")
            try:
                self._generate_canvas(script_text, width, height, image_path)
            except Exception as e:
                logger.error(f"[{job_id}] Visual canvas generation failed: {e}")
                raise RuntimeError(f"Canvas generation failed: {e}")

            # --- Step 3: FFmpeg Multiplexing and Rendering ---
            progress_callback(70, "Rendering final audio and video layers via FFmpeg...")
            local_output_mp4 = os.path.join(temp_dir, f"{job_id}_final.mp4")
            
            try:
                self._run_ffmpeg(
                    image_path=image_path,
                    audio_path=audio_path,
                    music_path=music_file,
                    output_path=local_output_mp4,
                    duration=duration,
                    fps=fps,
                    bitrate=bitrate
                )
            except Exception as e:
                logger.error(f"[{job_id}] FFmpeg rendering pipeline failed: {e}")
                raise RuntimeError(f"FFmpeg rendering failed: {e}")

            # --- Step 4: Cloudinary Upload ---
            progress_callback(90, "Uploading render assets to Cloudinary...")
            try:
                upload_result = cloudinary.uploader.upload_large(
                    local_output_mp4,
                    resource_type="video",
                    public_id=f"cabangile_studio/{job_id}",
                    overwrite=True
                )
                cloudinary_url = upload_result.get("secure_url")
                if not cloudinary_url:
                    raise KeyError("secure_url not found in Cloudinary response.")
                
                logger.info(f"[{job_id}] Successfully uploaded to Cloudinary: {cloudinary_url}")
                return cloudinary_url
                
            except Exception as e:
                logger.error(f"[{job_id}] Cloudinary upload failed: {e}")
                raise RuntimeError(f"Cloudinary hosting failed: {e}")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Uses ffprobe to extract exact duration of the generated audio track."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())

    def _generate_canvas(self, text: str, width: int, height: int, output_path: str):
        """Generates a clean background image frame with the script text burned in."""
        # Clean solid dark blue/grey aesthetic
        img = Image.new("RGB", (width, height), color="#1e293b")
        draw = ImageDraw.Draw(img)
        
        # Load a default system font
        try:
            font = ImageFont.load_default()
        except IOError:
            font = ImageFont.load_default()

        # Wrap script text to fit inside the video frame boundaries
        margin = 100
        max_width = width - (2 * margin)
        wrapped_text = self._wrap_text(text, font, max_width, draw)

        # Draw the text vertically centered
        text_y = height // 3
        for line in wrapped_text:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text(((width - text_w) // 2, text_y), line, font=font, fill="#f8fafc")
            text_y += text_h + 15

        img.save(output_path)

    def _wrap_text(self, text: str, font: ImageFont, max_width: int, draw: ImageDraw) -> list:
        """Helper to break lines so that they stay within the screen resolution width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def _run_ffmpeg(
        self, 
        image_path: str, 
        audio_path: str, 
        music_path: Optional[str], 
        output_path: str, 
        duration: float, 
        fps: int, 
        bitrate: str
    ):
        """Constructs and executes the FFmpeg command to merge voiceover, background music, and canvas."""
        # 1. Loop the single image over the duration of the audio
        # 2. Add the main TTS voiceover input
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(fps), "-t", f"{duration:.3f}", "-i", image_path,
            "-i", audio_path
        ]

        # 3. Handle optional background music mixing
        if music_path and os.path.exists(music_path):
            cmd.extend(["-stream_loop", "-1", "-i", music_path])
            # Mix voice track (volume=1.0) and music background (volume=0.15)
            filter_complex = (
                "[1:a]volume=1.0[voice];"
                f"[2:a]volume=0.15,atrim=duration={duration:.3f}[bg];"
                "[voice][bg]amix=inputs=2:duration=first[a]"
            )
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "0:v", "-map", "[a]"
            ])
        else:
            # Map default voice track straight to output
            cmd.extend(["-map", "0:v", "-map", "1:a"])

        # 4. Standard H.264 video compression parameters for web streaming compatibility
        cmd.extend([
            "-c:v", "libx264",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path
        ])

        logger.info(f"Executing FFmpeg Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")
