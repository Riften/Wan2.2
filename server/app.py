"""
FastAPI application for Wan2.2 video generation server.

Provides HTTP endpoints for submitting generation tasks, querying status,
and downloading results. The actual generation is handled by ModelWorker.
"""

import base64
import logging
import os
import uuid
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --------------- Pydantic models --------------- #


class GenerateRequest(BaseModel):
    """Request body for video generation."""

    prompt: str = Field(..., description="Text prompt for video generation")
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded input image (provide this OR image_path)"
    )
    image_path: Optional[str] = Field(
        None,
        description="Path to input image on the server filesystem",
    )
    size: str = Field(
        "480*832",
        description="Video resolution as width*height. Supported: 720*1280, 1280*720, 480*832, 832*480",
    )
    frame_num: Optional[int] = Field(
        None, description="Number of frames (should be 4n+1). Default from config."
    )
    shift: Optional[float] = Field(
        None, description="Noise schedule shift. Default from config."
    )
    sample_solver: str = Field("unipc", description="Solver: unipc or dpm++")
    sampling_steps: Optional[int] = Field(
        None, description="Diffusion sampling steps. Default from config."
    )
    guide_scale: Optional[Union[float, List[float]]] = Field(
        None,
        description="Classifier-free guidance scale. Float or [low_noise, high_noise].",
    )
    seed: int = Field(-1, description="Random seed (-1 for random)")
    save_path: Optional[str] = Field(
        None,
        description="Server filesystem path to save the output video. "
        "If provided, the video is written directly to this path. "
        "Useful when client and server share a filesystem.",
    )


class GenerateResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    queue_position: int = 0
    prompt: str = ""
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_path: Optional[str] = None


class QueueStatusResponse(BaseModel):
    pending: int
    processing: int
    completed: int
    failed: int
    total: int


# --------------- App factory --------------- #

# Module-level worker reference, set by run.py before starting uvicorn
worker = None


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    app = FastAPI(
        title="Wan2.2 Video Generation Server",
        description="HTTP API for Wan2.2 image-to-video generation with multi-GPU support",
        version="1.0.0",
    )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        """Submit a video generation task. Returns a task_id for tracking."""
        if worker is None:
            raise HTTPException(status_code=503, detail="Model worker not initialized")

        if req.image_base64 is None and req.image_path is None:
            raise HTTPException(
                status_code=400,
                detail="Either image_base64 or image_path must be provided",
            )

        # Handle image upload
        temp_image = False
        if req.image_base64 is not None:
            try:
                img_data = base64.b64decode(req.image_base64)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
            img_path = os.path.join("temp_images", f"{uuid.uuid4().hex}.png")
            with open(img_path, "wb") as f:
                f.write(img_data)
            temp_image = True
        else:
            img_path = req.image_path
            if not os.path.isfile(img_path):
                raise HTTPException(
                    status_code=400, detail=f"Image file not found: {img_path}"
                )

        # Build params dict
        params = {
            "prompt": req.prompt,
            "image_path": img_path,
            "size": req.size,
            "frame_num": req.frame_num,
            "shift": req.shift,
            "sample_solver": req.sample_solver,
            "sampling_steps": req.sampling_steps,
            "guide_scale": req.guide_scale,
            "seed": req.seed,
            "_temp_image": temp_image,
            "save_path": req.save_path,
        }

        task_id = worker.submit_task(params)
        return GenerateResponse(task_id=task_id, status="queued")

    @app.get("/task/{task_id}", response_model=TaskStatusResponse)
    async def get_task(task_id: str):
        """Query the status of a generation task."""
        if worker is None:
            raise HTTPException(status_code=503, detail="Model worker not initialized")
        info = worker.get_task_info(task_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return info

    @app.get("/task/{task_id}/video")
    async def get_video(task_id: str):
        """Download the generated video file (only available after task completion)."""
        if worker is None:
            raise HTTPException(status_code=503, detail="Model worker not initialized")
        info = worker.get_task_info(task_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if info["status"] != "completed":
            current_status = info["status"]
            raise HTTPException(
                status_code=400,
                detail=f"Task status is {current_status!r}, video not yet available",
            )
        path = worker.get_video_path(task_id)
        if path is None or not os.path.isfile(path):
            raise HTTPException(status_code=500, detail="Video file missing")
        return FileResponse(
            path, media_type="video/mp4", filename=f"{task_id}.mp4"
        )

    @app.get("/queue/status", response_model=QueueStatusResponse)
    async def queue_status():
        """Get overall task queue statistics."""
        if worker is None:
            raise HTTPException(status_code=503, detail="Model worker not initialized")
        return worker.get_queue_status()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "model_loaded": worker is not None}

    return app
