"""Cabangile AI Text-to-Video Engine Main Entry Point.

Combines the FastAPI backend engine with static file hosting to launch
the fully integrated video generation ecosystem on a unified server port.
"""

import logging
import os
from pathlib import Path
import shutil
import uvicorn

# Setup localized logging layout configuration profiles
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CabangileMain")


def verify_environment() -> None:
    """Validates that necessary directories and native system tool pipelines exist."""
    logger.info("Verifying system execution environment profiles...")
    
    # 1. Ensure basic output directories exist
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("static").mkdir(parents=True, exist_ok=True)

    # 2. Check for dependencies on local PATH
    for tool in ["ffmpeg", "ffprobe"]:
        if not shutil.which(tool):
            logger.warning(
                f"Core binary resource '{tool}' was not detected on your system PATH environment. "
                f"Video generation steps will fail unless '{tool}' is correctly installed."
            )
            
    # 3. Ensure index.html exists inside static folder for hosting stability
    source_html = Path("index.html")
    target_html = Path("static/index.html")
    
    if source_html.exists() and not target_html.exists():
        shutil.copy(source_html, target_html)
        logger.info("Migrated root index.html template context layer into static assets target.")
    elif not target_html.exists():
        # Fallback empty placeholder to keep ASGI mount layers safe if index.html is missing
        with open(target_html, "w", encoding="utf-8") as f:
            f.write("<h1>Cabangile Studio Static Workspace Host Pipeline Mount Active.</h1>")


# We import the app instance dynamically or wrap it securely 
# to allow verification routines to prepare file buffers first.
from fastapi.staticfiles import StaticFiles
from api_server import app

# Mount static files folder to serve the UI seamlessly at the root URL
verify_environment()
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    logger.info("Launching integrated production pipeline server cluster layout...")

    # Render provides the PORT environment variable.
    # Use port 8000 when running locally.
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
    )
