import os
import uuid
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Cloudinary SDK imports
import cloudinary
import cloudinary.uploader

# Import the video engine module
from video_engine import VideoEngine

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CabangileVideoAPI")

# ==========================================
# CLOUDINARY CONFIGURATION
# ==========================================
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

# Conditional initialization to prevent errors if keys are absent
CLOUDINARY_READY = False
if all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
    CLOUDINARY_READY = True
    logger.info("Cloudinary CDN configuration successfully initialized.")
else:
    logger.warning(
        "Cloudinary environment variables are missing! "
        "Uploads will be skipped, falling back to local file storage only."
    )

# ==========================================
# FASTAPI & CORSMIDDLEWARE SETUP
# ==========================================
app = FastAPI(
    title="Cabangile AI Video Studio API",
    description="Production API server for asynchronous AI video generation and Cloudinary hosting.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# IN-MEMORY DATABASE
# ==========================================
JOBS_DB: Dict[str, Dict[str, Any]] = {}

# ==========================================
# PYDANTIC SCHEMAS (REQUEST / RESPONSE)
# ==========================================
class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt or script for the video generation.")
    bg_music_path: Optional[str] = Field(None, description="Optional path to a background music file.")

class VideoGenerationResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the asynchronous generation task.")
    status: str = Field(..., description="Current status of the job (pending, processing).")
    message: str = Field(..., description="Initial status confirmation message.")

class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="Current status: pending, processing, completed, failed.")
    progress_message: str = Field(..., description="Human-readable updates regarding the engine status.")
    result: Optional[Dict[str, Any]] = Field(None, description="Output payload containing file paths and asset URLs.")

# ==========================================
# BACKGROUND WORKER TASK
# ==========================================
def background_video_render(job_id: str, prompt: str, bg_music_path: Optional[str] = None) -> None:
    """
    Executes the VideoEngine generation loop, performs safety validations,
    attempts to upload the final asset to Cloudinary, and cleans up local storage.
    """
    logger.info(f"Starting background job {job_id}")
    JOBS_DB[job_id]["status"] = "processing"
    JOBS_DB[job_id]["progress_message"] = "Initializing VideoEngine pipeline..."

    local_output_path = None

    try:
        # 1. Trigger Video Generation via Engine
        engine = VideoEngine()
        JOBS_DB[job_id]["progress_message"] = "Rendering video frames and assembling audio..."
        
        engine_result = engine.generate(
            text_script=prompt,
            bg_music_path=bg_music_path,
            progress_callback=None
        )

        # 2. Parse Engine Output Dictionary
        if not engine_result or not engine_result.get("success"):
            errors = engine_result.get("errors", ["Unknown engine processing error"])
            raise RuntimeError(f"VideoEngine failed pipeline generation: {', '.join(errors)}")

        local_output_path = engine_result.get("output_path")

        # Verify Local Output exists on disk
        if not local_output_path or not os.path.exists(local_output_path):
            raise FileNotFoundError(f"VideoEngine reported success, but file was missing at: {local_output_path}")

        logger.info(f"Job {job_id} successfully rendered locally at {local_output_path}")
        
        # Prepare the base completion state
        JOBS_DB[job_id]["status"] = "completed"
        JOBS_DB[job_id]["result"] = {
            "local_path": local_output_path,
            "cloudinary_url": None,
            "upload_error": None,
            "metadata": {
                "duration": engine_result.get("duration"),
                "total_scenes": engine_result.get("total_scenes"),
                "render_time": engine_result.get("render_time")
            }
        }

        # 3. Cloudinary Upload & Disk Cleanup Logic
        if CLOUDINARY_READY:
            JOBS_DB[job_id]["progress_message"] = "Video rendered. Uploading to Cloudinary CDN..."
            try:
                upload_result = cloudinary.uploader.upload(
                    local_output_path,
                    public_id=f"cabangile_job_{job_id}",
                    resource_type="video",
                    overwrite=True
                )
                secure_url = upload_result.get("secure_url")
                JOBS_DB[job_id]["result"]["cloudinary_url"] = secure_url
                JOBS_DB[job_id]["progress_message"] = "Job completed successfully. Asset fully hosted."
                logger.info(f"Job {job_id} successfully hosted on Cloudinary: {secure_url}")
                
                # Cleanup disk space now that CDN safely holds the file
                try:
                    os.remove(local_output_path)
                    JOBS_DB[job_id]["result"]["local_path"] = None  # Clear path since file is deleted
                    logger.info(f"Cleaned up local file to preserve space: {local_output_path}")
                except OSError as os_err:
                    logger.warning(f"Failed to remove local file {local_output_path}: {str(os_err)}")

            except Exception as cloud_err:
                error_msg = f"Cloudinary upload failed: {str(cloud_err)}"
                logger.error(f"Job {job_id} completed rendering but failed CDN upload: {error_msg}")
                JOBS_DB[job_id]["result"]["upload_error"] = error_msg
                JOBS_DB[job_id]["progress_message"] = (
                    "Job completed with upload errors. File retained locally for direct download."
                )
        else:
            JOBS_DB[job_id]["progress_message"] = (
                "Job completed. Cloudinary skipped (missing configs). File available via direct download."
            )

    except Exception as exc:
        logger.error(f"Fatal error executing job {job_id}: {str(exc)}", exc_info=True)
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["progress_message"] = f"Generation failed: {str(exc)}"
        JOBS_DB[job_id]["result"] = {"error_details": str(exc)}

# ==========================================
# REST ENDPOINTS
# ==========================================
@app.post(
    "/api/video/generate", 
    response_model=VideoGenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger asynchronous video generation"
)
async def generate_video(payload: VideoGenerationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    
    JOBS_DB[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress_message": "Job queued. Waiting to spin up engine thread.",
        "result": None
    }
    
    background_tasks.add_task(
        background_video_render,
        job_id=job_id,
        prompt=payload.prompt,
        bg_music_path=payload.bg_music_path
    )
    
    logger.info(f"Accepted video generation request. Job ID assigned: {job_id}")
    return VideoGenerationResponse(
        job_id=job_id,
        status="pending",
        message="Video generation has been successfully offloaded to background threads."
    )

@app.get(
    "/api/video/status/{job_id}", 
    response_model=JobStatusResponse,
    summary="Check video generation job status"
)
async def get_job_status(job_id: str):
    job = JOBS_DB.get(job_id)
    if not job:
        logger.warning(f"Status requested for non-existent job ID: {job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job sequence reference '{job_id}' could not be located."
        )
    return job

@app.get(
    "/api/video/download/{job_id}", 
    summary="Download rendered video asset from local disk"
)
async def download_video(job_id: str):
    job = JOBS_DB.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Job sequence reference could not be located."
        )
        
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Resource is not ready for retrieval. Current state is: {job.get('status')}"
        )
        
    result = job.get("result", {})
    local_path = result.get("local_path")
    
    if not local_path or not os.path.exists(local_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="The requested file asset is no longer stored locally on this instance."
        )
        
    logger.info(f"Serving local binary file stream for job {job_id}")
    return FileResponse(
        path=local_path, 
        media_type="video/mp4", 
        filename=f"cabangile_{job_id}.mp4"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Bootstrapping Cabangile AI Video Studio API on port {port}")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
