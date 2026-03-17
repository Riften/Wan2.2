"""
Example: Using the Wan2.2 video generation client.

Usage:
    python client/example.py --server http://localhost:8000 \
        --image examples/i2v_input.JPG \
        --prompt "A white cat surfing on the beach."

This script demonstrates both the step-by-step API and the
convenience method generate_and_download().
"""

import argparse
import logging
import os
import sys
import time

# Allow running as  python client/example.py  from the project root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from client.wan_client import WanClient

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def demo_step_by_step(client: WanClient, args):
    """Low-level API: submit -> poll -> download."""
    logger.info("=== Step-by-step demo ===")

    # 1. Check server health
    health = client.health()
    logger.info(f"Server health: {health}")

    # 2. Check queue status
    queue = client.get_queue_status()
    logger.info(f"Queue status: {queue}")

    # 3. Submit generation task
    task_id = client.generate(
        prompt=args.prompt,
        image_path=args.image,
        size=args.size,
        seed=args.seed,
    )
    logger.info(f"Submitted task: {task_id}")

    # 4. Poll until completion
    while True:
        status = client.get_task_status(task_id)
        logger.info(
            f"Task {task_id}: {status['status']} "
            f"(queue_position={status.get('queue_position', 0)})"
        )
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(5)

    if status["status"] == "failed":
        logger.error(f"Generation failed: {status.get('error')}")
        return

    # 5. Download the video
    save_path = args.output or f"{task_id}.mp4"
    client.download_video(task_id, save_path)
    logger.info(f"Video downloaded to: {save_path}")


def demo_convenience(client: WanClient, args):
    """High-level API: one call does everything."""
    logger.info("=== Convenience demo ===")

    save_path = args.output or "output_convenience.mp4"
    result = client.generate_and_download(
        prompt=args.prompt,
        image_path=args.image,
        save_path=save_path,
        size=args.size,
        seed=args.seed,
        poll_interval=5,
        timeout=1800,
    )
    logger.info(f"Done! Video saved to: {result}")


def demo_server_image(client: WanClient, args):
    """Use an image already on the server filesystem."""
    logger.info("=== Server-side image demo ===")

    save_path = args.output or "output_server_image.mp4"
    result = client.generate_and_download(
        prompt=args.prompt,
        server_image_path=args.server_image,
        save_path=save_path,
        size=args.size,
        seed=args.seed,
    )
    logger.info(f"Done! Video saved to: {result}")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Client Example")
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server URL",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Local image file to upload",
    )
    parser.add_argument(
        "--server_image",
        type=str,
        default=None,
        help="Image path on the server (e.g. examples/i2v_input.JPG)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
    )
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["step", "convenience", "server_image"],
        default="convenience",
        help="Demo mode to run",
    )
    args = parser.parse_args()

    client = WanClient(base_url=args.server)

    if args.mode == "step":
        if args.image is None and args.server_image is None:
            parser.error("--image or --server_image is required for step mode")
        demo_step_by_step(client, args)
    elif args.mode == "convenience":
        if args.image is None:
            parser.error("--image is required for convenience mode")
        demo_convenience(client, args)
    elif args.mode == "server_image":
        if args.server_image is None:
            parser.error("--server_image is required for server_image mode")
        demo_server_image(client, args)


if __name__ == "__main__":
    main()
