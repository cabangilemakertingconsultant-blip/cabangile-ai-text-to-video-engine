"""Cabangile AI Video Generation API Server.

Exposes the VideoEngine pipeline via a RESTful API with support for
asynchronous background rendering tasks and job status tracking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import the existing pipeline components
# Ensure your engine script is named 'video_engine.py' or update this import accordingly
from video_engine import VideoEngine

# Setup Logging
logger = logging.getLogger("CabangileAPI")

app = FastAPI(
    title="Cabangile AI Video API",
    description="Backend API server for managing text-to-video processing pipelines.",
    version="1.0.0"
)

# Enable CORS for frontend integration (e.g., live index.html testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory database to store job state updates
# For production, replace this with Redis or a proper SQL database
JOBS_DB: Dict[str, Dict] = {}


# --- Request/Response Schemas ---

class VideoRequest(BaseModel):
    script: str = Field(..., description="The double-newline separated text script for the scenes.")
    bg_music_path: Optional[str] = Field(None, description="Absolute path to an optional background audio file.")
    fps: int = Field(30, ge=15, le=60, description="Frames per second for output video compilation.")
    voice_language: str = Field("en", description="gTTS ISO 639-1 language matching layout profile.")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_message: str
    result: Optional[dict] = None


# --- Background Worker Core Task ---

def run_video_generation(job_id: str, request_data: VideoRequest):
    """Executes the VideoEngine generation loop inside an isolated background worker thread."""
    JOBS_DB[job_id]["status"] = "processing"
    
    def api_progress_callback(message: str) -> None:
        logger.info(f"[Job {job_id}]: {message}")
        JOBS_DB[job_id]["progress_message"] = message

    try:
        engine = VideoEngine(config={
            "fps": request_data.fps,
            "voice_language": request_data.voice_language,
            "output_directory": "output"
        })
        
        # Execute structural video compilation
        results = engine.generate(
            text_script=request_data.script,
            bg_music_path=request_data.bg_music_path,
            progress_callback=api_progress_callback
        )
        
        if results.get("success"):
            JOBS_DB[job_id]["status"] = "completed"
            JOBS_DB[job_id]["progress_message"] = "Video generated successfully!"
            JOBS_DB[job_id]["result"] = results
        else:
            JOBS_DB[job_id]["status"] = "failed"
            JOBS_DB[job_id]["progress_message"] = "Pipeline execution error."
            JOBS_DB[job_id]["result"] = results
            
    except Exception as e:
        logger.error(f"Background task failure for job {job_id}: {e}", exc_info=True)
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["progress_message"] = f"Fatal pipeline crash: {str(e)}"


# --- REST API Routing Endpoints ---

@app.post("/api/video/generate", status_code=status.HTTP_202_ACCEPTED, response_model=JobStatusResponse)
async def generate_video(payload: VideoRequest, background_tasks: BackgroundTasks):
    """Submits a script processing job payload into the asynchronous generation worker pool."""
    job_id = f"task_{uuid4().hex[:12]}"
    
    # Initialize state configuration layout footprint inside volatile dictionary storage structures
    JOBS_DB[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_message": "Job successfully received and queued.",
        "result": None
    }
    
    # Push the pipeline runtime target frame context loop array cleanly onto FastAPI background tasks threads
    background_tasks.add_task(run_video_generation, job_id, payload)
    
    return JOBS_DB[job_id]


@app.get("/api/video/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Retrieves current processing operational execution statuses for tracked engine pipelines."""
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Requested generation task target profile index not found.")
    return JOBS_DB[job_id]


@app.get("/api/video/download/{job_id}")
async def download_video(job_id: str):
    """Streams compiled static output MP4 binary video assets back down out of the host filesystem blocks."""
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job entry signature record invalid.")
        
    job = JOBS_DB[job_id]
    if job["status"] != "completed" or not job["result"]:
        raise HTTPException(status_code=400, detail="The requested media output pipeline target is not compiled.")
        
    video_file_path = job["result"].get("output_path")
    if not video_file_path or not Path(video_file_path).exists():
        raise HTTPException(status_code=404, detail="Target binary file container missing off physical disk volumes.")
        
    return FileResponse(
        path=video_file_path, 
        media_type="video/mp4", 
        filename=Path(video_file_path).name
    )


if __name__ == "__main__":
    import uvicorn
    # Start ASGI application servers on generic developer loop ports mapping targets
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
