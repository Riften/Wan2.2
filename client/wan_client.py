"""
Client library for the Wan2.2 video generation server.

Usage:
    from client import WanClient

    client = WanClient("http://server-host:8000")
    video_path = client.generate_and_download(
        prompt="A cat surfing on the beach.",
        image_path="input.jpg",
        save_path="output.mp4",
    )
"""

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


class WanClient:
    """
    HTTP client for the Wan2.2 video generation server.

    Provides both low-level methods (submit, poll, download) and a
    high-level convenience method (generate_and_download) that handles
    the full lifecycle.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Args:
            base_url: Server URL, e.g. "http://10.0.0.1:8000".
            timeout: HTTP request timeout in seconds (for non-download requests).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------ #
    #  Low-level API
    # ------------------------------------------------------------------ #

    def health(self) -> dict:
        """Check server health."""
        r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        server_image_path: Optional[str] = None,
        server_save_path: Optional[str] = None,
        size: str = "480*832",
        frame_num: Optional[int] = None,
        shift: Optional[float] = None,
        sample_solver: str = "unipc",
        sampling_steps: Optional[int] = None,
        guide_scale: Optional[Union[float, List[float]]] = None,
        seed: int = -1,
    ) -> str:
        """
        Submit a generation task and return the task_id.

        Provide exactly one image source:
          - image_path: local file path (will be base64-encoded and uploaded)
          - image_base64: pre-encoded base64 string
          - server_image_path: path to an image already on the server

        Optionally specify output location:
          - server_save_path: path on the server filesystem to save the video.
            Useful when client and server share storage (same machine or
            shared mount), avoiding the need to download the video.

        Returns:
            task_id (str)
        """
        # Resolve image
        if image_path is not None:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "size": size,
            "sample_solver": sample_solver,
            "seed": seed,
        }
        if image_base64 is not None:
            payload["image_base64"] = image_base64
        elif server_image_path is not None:
            payload["image_path"] = server_image_path
        else:
            raise ValueError(
                "Provide one of: image_path, image_base64, or server_image_path"
            )

        if frame_num is not None:
            payload["frame_num"] = frame_num
        if shift is not None:
            payload["shift"] = shift
        if sampling_steps is not None:
            payload["sampling_steps"] = sampling_steps
        if guide_scale is not None:
            payload["guide_scale"] = guide_scale
        if server_save_path is not None:
            payload["save_path"] = server_save_path

        r = self.session.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["task_id"]

    def get_task_status(self, task_id: str) -> dict:
        """Query the current status of a task."""
        r = self.session.get(
            f"{self.base_url}/task/{task_id}", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def get_queue_status(self) -> dict:
        """Get the overall queue statistics."""
        r = self.session.get(
            f"{self.base_url}/queue/status", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def download_video(self, task_id: str, save_path: str) -> str:
        """
        Download the generated video to a local file.

        Args:
            task_id: The task identifier.
            save_path: Local path to save the video.

        Returns:
            The save_path on success.

        Raises:
            requests.HTTPError if video is not ready or task not found.
        """
        r = self.session.get(
            f"{self.base_url}/task/{task_id}/video",
            timeout=600,  # large timeout for video download
            stream=True,
        )
        r.raise_for_status()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

    # ------------------------------------------------------------------ #
    #  High-level API
    # ------------------------------------------------------------------ #

    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 3.0,
        timeout: float = 1800,
    ) -> dict:
        """
        Poll a task until it reaches a terminal state (completed / failed).

        Args:
            task_id: The task identifier.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Returns:
            Final task status dict.
        """
        start = time.time()
        last_status = None
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task_id} did not complete within {timeout}s"
                )
            status = self.get_task_status(task_id)
            if status["status"] != last_status:
                last_status = status["status"]
                pos_info = ""
                if status.get("queue_position", 0) > 0:
                    pos_info = f" (queue position: {status['queue_position']})"
                logger.info(
                    f"Task {task_id}: {status['status']}{pos_info} "
                    f"[{elapsed:.0f}s elapsed]"
                )
            if status["status"] in ("completed", "failed"):
                return status
            time.sleep(poll_interval)

    def generate_and_download(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        server_image_path: Optional[str] = None,
        server_save_path: Optional[str] = None,
        save_path: Optional[str] = None,
        poll_interval: float = 3.0,
        timeout: float = 1800,
        **gen_kwargs,
    ) -> str:
        """
        Full pipeline: submit -> wait -> download.

        Args:
            prompt: Text prompt.
            image_path: Local image file to upload.
            image_base64: Pre-encoded image.
            server_image_path: Image path on the server.
            server_save_path: Path on the server to save the output video.
                       When set, the server writes the video directly to
                       this path. If client and server share storage,
                       download is skipped automatically.
            save_path: Where to save the video locally (via download).
                       If None, saves to "<task_id>.mp4" in cwd.
                       Ignored when server_save_path is set.
            poll_interval: Seconds between polls.
            timeout: Max wait time in seconds.
            **gen_kwargs: Additional generation parameters
                          (size, frame_num, shift, etc.)

        Returns:
            Local path of the downloaded video.

        Raises:
            RuntimeError: If generation fails.
            TimeoutError: If generation times out.
        """
        task_id = self.generate(
            prompt=prompt,
            image_path=image_path,
            image_base64=image_base64,
            server_image_path=server_image_path,
            server_save_path=server_save_path,
            **gen_kwargs,
        )
        logger.info(f"Submitted task {task_id}")

        result = self.wait_for_completion(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

        if result["status"] == "failed":
            raise RuntimeError(
                f"Generation failed for task {task_id}: {result.get('error', 'unknown')}"
            )

        # When server_save_path is set, the video is already at the
        # requested location on the server/shared filesystem -- skip download.
        if server_save_path is not None:
            result_path = result.get("result_path", server_save_path)
            logger.info(f"Video saved on server at {result_path}")
            return result_path

        if save_path is None:
            save_path = f"{task_id}.mp4"

        self.download_video(task_id, save_path)
        logger.info(f"Video saved to {save_path}")
        return save_path
