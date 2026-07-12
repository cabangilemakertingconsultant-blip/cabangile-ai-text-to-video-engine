"""Cabangile AI Video Studio - API Server.

This module provides a production-grade FastAPI server for managing asynchronous
video generation jobs, handling local processing, Cloudinary storage, 
and file downloads.
"""

import os
import uuid
import logging
from typing import Dict, Any, Optional
import time

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
import cloudinary
import cloudinary.uploader

# Importing VideoEngine from video_engine.py
from video_engine import VideoEngine

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api_server")

# -----------------------------------------------------------------------------
# CLOUDINARY CONFIGURATION
# -----------------------------------------------------------------------------
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

CLOUDINARY_ENABLED = False

if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    try:
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET,
            secure=True
        )
        CLOUDINARY_ENABLED = True
        logger.info("Cloudinary successfully configured and enabled.")
    except Exception as e:
        logger.error(f"Failed to initialize Cloudinary despite available env vars: {e}")
else:
    logger.warning("Cloudinary credentials missing. Video uploads will be skipped, and files kept locally.")

# -----------------------------------------------------------------------------
# FASTAPI APP & IN-MEMORY STORAGE
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Cabangile AI Video Studio API",
    description="Production API server for managing asynchronous AI video generation pipelines.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory dictionary tracking asynchronous video generation jobs
JOBS_DB: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# API MODELS
# -----------------------------------------------------------------------------
class VideoGenerationRequest(BaseModel):
    """Schema for a video generation request."""
    script: str = Field(..., min_length=1, description="The textual script for video generation.")
    bg_music_path: Optional[str] = Field(None, description="Optional local file path to background music.")
    voice_language: str = Field("en", description="ISO language code for the voice synthesis.")
    fps: int = Field(default=30, ge=15, le=60, description="Frames per second. Must be between 15 and 60.")

    class Config:
        json_schema_extra = {
            "example": {
                "script": "Welcome to Cabangile AI Video Studio. Creating content instantly.",
                "bg_music_path": "/assets/music/ambient.mp3",
                "voice_language": "en",
                "fps": 30
            }
        }

class VideoGenerationResponse(BaseModel):
    """Schema for video generation request acknowledgment."""
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    """Schema for checking the status of an active or historical job."""
    job_id: str
    status: str
    progress_message: str
    result: Optional[Dict[str, Any]] = None

# -----------------------------------------------------------------------------
# BACKGROUND WORKER
# -----------------------------------------------------------------------------
def background_video_render(
    job_id: str,
    script: str,
    bg_music_path: Optional[str],
    voice_language: str,
    fps: int
) -> None:
    """Processes video generation asynchronously in the background.

    Args:
        job_id (str): Unique tracking identifier for the job.
        script (str): Text script used for rendering.
        bg_music_path (Optional[str]): Audio file path for background soundtrack.
        voice_language (str): Voice translation/synthesis language code.
        fps (int): Frame rate execution constraint.
    """
    logger.info(f"Starting background job {job_id}...")
    start_time = time.time()
    
    # Improvement 1: Use .update() to preserve and build upon the entry state safely
    JOBS_DB[job_id].update({
        "status": "processing",
        "progress_message": "Initializing VideoEngine and rendering pipeline...",
        "result": None
    })

    try:
        engine = VideoEngine()
        
        # Invoke generation framework and capture dictionary result safely
        # Note: Ensure video_engine.py signature supports voice_language and fps!
        engine_result = engine.generate(
            text_script=script,
            bg_music_path=bg_music_path,
            voice_language=voice_language,
            fps=fps,
            progress_callback=None
        )

        if not engine_result or not engine_result.get("success"):
            raise RuntimeError("Video generation failed or returned an unsuccessful status.")

        output_path = engine_result.get("output_path")
        
        if not output_path or not os.path.exists(output_path):
            raise FileNotFoundError(f"Generated output file not found at path: {output_path}")

        # Compute render time if engine doesn't supply it
        fallback_render_time = round(time.time() - start_time, 2)
        render_time = engine_result.get("render_time", fallback_render_time)

        metadata = {
            "output_path": output_path,
            "render_time": render_time,
            "duration": engine_result.get("duration", 0.0),
            "total_scenes": engine_result.get("total_scenes", 0),
            "cloudinary_url": None
        }

        # Handle Cloudinary Integration if credentials exist
        if CLOUDINARY_ENABLED:
            JOBS_DB[job_id].update({
                "progress_message": "Uploading generated video to Cloudinary..."
            })
            logger.info(f"Uploading file {output_path} to Cloudinary for job {job_id}")
            
            upload_result = cloudinary.uploader.upload(
                output_path, 
                resource_type="video",
                folder="cabangile_studio"
            )
            
            secure_url = upload_result.get("secure_url")
            metadata["cloudinary_url"] = secure_url
            
            # Delete local file safely after successful upload
            if secure_url and os.path.exists(output_path):
                os.remove(output_path)
                logger.info(f"Successfully deleted local file {output_path} after upload.")
                metadata["output_path"] = None  # Reference dropped locally

        JOBS_DB[job_id].update({
            "status": "completed",
            "progress_message": "Video generated and finalized successfully.",
            "result": metadata
        })
        logger.info(f"Job {job_id} finalized successfully.")

    except Exception as e:
        logger.error(f"Error handling background execution for job {job_id}: {str(e)}", exc_info=True)
        JOBS_DB[job_id].update({
            "status": "failed",
            "progress_message": f"Job failed during compilation: {str(e)}",
            "result": None
        })

# -----------------------------------------------------------------------------
# REST ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", summary="Root Health Check")
def root() -> Dict[str, str]:
    """Returns api runtime connectivity credentials."""
    return {
        "name": "Cabangile AI Video Studio API",
        "version": "2.0.0",
        "status": "running"
    }

@app.post(
    "/api/video/generate",
    response_model=VideoGenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Asynchronous Video Generation Process"
)
def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks
) -> VideoGenerationResponse:
    """Enqueues a text-to-video generation request."""
    # Input validation for empty whitespace scripts
    if not request.script.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The input script cannot be blank or whitespace."
        )

    job_id = str(uuid.uuid4())
    
    # Register job initial status explicitly, including the job_id
    JOBS_DB[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_message": "Job queued in processing framework.",
        "result": None
    }

    # Dispatch to asynchronous background tasks executor
    background_tasks.add_task(
        background_video_render,
        job_id=job_id,
        script=request.script,
        bg_music_path=request.bg_music_path,
        voice_language=request.voice_language,
        fps=request.fps
    )

    return VideoGenerationResponse(
        job_id=job_id,
        status="queued",
        message="Video generation job accepted and successfully pushed to background worker pipeline."
    )

@app.get(
    "/api/video/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Fetch Video Generation Status and Details"
)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Fetches full state configurations metrics tied to processing records."""
    if job_id not in JOBS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job record associated with identifier '{job_id}' could not be resolved."
        )
    
    job_info = JOBS_DB[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job_info["status"],
        progress_message=job_info["progress_message"],
        result=job_info["result"]
    )

@app.get(
    "/api/video/download/{job_id}",
    summary="Download Generated MP4 Video File"
)
def download_video(job_id: str) -> Any:
    """Downloads local video payload or redirects seamlessly to Cloudinary storage."""
    if job_id not in JOBS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job record associated with identifier '{job_id}' could not be resolved."
        )

    job_info = JOBS_DB[job_id]

    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job processing incomplete. Current status profile: '{job_info['status']}'."
        )

    result_metadata = job_info.get("result") or {}
    local_path = result_metadata.get("output_path")

    # Improvement 2: Seamless handling of Cloudinary cloud storage via HTTP Redirects
    if not local_path or not os.path.exists(local_path):
        cloudinary_url = result_metadata.get("cloudinary_url")
        if cloudinary_url:
            logger.info(f"Redirecting download request for job {job_id} to Cloudinary.")
            return RedirectResponse(url=cloudinary_url, status_code=status.HTTP_303_SEE_OTHER)
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The local video artifact could not be found on disk."
        )

    return FileResponse(
        path=local_path,
        media_type="video/mp4",
        filename=f"video_{job_id}.mp4"
    )

# -----------------------------------------------------------------------------
# SERVER EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port_number = int(os.getenv("PORT", 8000))
    logger.info(f"Starting ASGI Production server setup binding initialization on port: {port_number}")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port_number,
        reload=False
    )
