"""
Model worker module for Wan2.2 video generation server.

Handles model initialization, task queue management, and multi-GPU coordination.
All ranks participate in the work loop; only rank 0 manages the task queue
and runs the HTTP server.
"""

import logging
import os
import queue
import random
import sys
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video

logger = logging.getLogger(__name__)


class ModelWorker:
    """
    Manages Wan2.2 model lifecycle and task processing across distributed GPU workers.

    Architecture:
        - Rank 0: runs HTTP server (background thread) + coordinates work
        - All ranks: participate in generation via synchronized work loop
        - Task queue: thread-safe FIFO queue for sequential generation
    """

    def __init__(self, args):
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = self.local_rank
        self.output_dir = getattr(args, "output_dir", "outputs")
        self.task_type = getattr(args, "task", "i2v-A14B")
        self._shutdown = False

        # Setup logging
        if self.rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)],
                force=True,
            )
        else:
            logging.basicConfig(level=logging.ERROR, force=True)

        # Init distributed
        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
            )
            logger.info(
                f"Rank {self.rank}/{self.world_size} initialized on GPU {self.local_rank}."
            )

        if getattr(args, "ulysses_size", 1) > 1:
            assert args.ulysses_size == self.world_size, (
                f"ulysses_size ({args.ulysses_size}) must equal "
                f"world_size ({self.world_size})"
            )
            from wan.distributed.util import init_distributed_group

            init_distributed_group()

        # Init model
        self._init_model(args)

        # Task management (rank 0 only)
        if self.rank == 0:
            self.task_queue = queue.Queue()
            self.tasks: Dict[str, Dict[str, Any]] = OrderedDict()
            self.lock = threading.Lock()
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs("temp_images", exist_ok=True)
            logger.info(f"Model worker ready. Output dir: {self.output_dir}")

    def _init_model(self, args):
        """Initialize the generation model based on task type."""
        cfg = WAN_CONFIGS[self.task_type]
        self.cfg = cfg
        use_sp = getattr(args, "ulysses_size", 1) > 1
        t5_fsdp = getattr(args, "t5_fsdp", False)
        dit_fsdp = getattr(args, "dit_fsdp", False)
        t5_cpu = getattr(args, "t5_cpu", False)

        # Determine model class based on task type (mirrors generate.py logic)
        if "t2v" in self.task_type:
            logger.info("Creating WanT2V pipeline...")
            self.model = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=self.device,
                rank=self.rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_sp=use_sp,
                t5_cpu=t5_cpu,
            )
        else:
            # Default: i2v-A14B
            logger.info("Creating WanI2V pipeline...")
            self.model = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=self.device,
                rank=self.rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_sp=use_sp,
                t5_cpu=t5_cpu,
            )

        logger.info(f"Model loaded on rank {self.rank}.")

    # ------------------------------------------------------------------ #
    #  Task management (rank 0 only)
    # ------------------------------------------------------------------ #

    def submit_task(self, params: dict) -> str:
        """Enqueue a generation task. Returns a unique task_id."""
        task_id = uuid.uuid4().hex[:8]
        task_info = {
            "task_id": task_id,
            "status": "queued",
            "params": params,
            "result_path": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        with self.lock:
            self.tasks[task_id] = task_info
            self.task_queue.put(task_id)
            logger.info(
                f"Task {task_id} queued. Queue size: {self.task_queue.qsize()}"
            )
        return task_id

    def get_task_info(self, task_id: str) -> Optional[dict]:
        """Return public info for a task (no internal params exposed)."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            info = {
                "task_id": task["task_id"],
                "status": task["status"],
                "error": task["error"],
                "created_at": task["created_at"],
                "completed_at": task["completed_at"],
                "prompt": task["params"].get("prompt", ""),
                "result_path": task["result_path"],
            }
            # Calculate queue position
            if task["status"] == "queued":
                pos = 0
                for tid in list(self.task_queue.queue):
                    pos += 1
                    if tid == task_id:
                        break
                info["queue_position"] = pos
            else:
                info["queue_position"] = 0
            return info

    def get_video_path(self, task_id: str) -> Optional[str]:
        """Return the file path if the task is completed, else None."""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task["status"] == "completed" and task["result_path"]:
                return task["result_path"]
            return None

    def get_queue_status(self) -> dict:
        """Return aggregate queue statistics."""
        with self.lock:
            pending = self.task_queue.qsize()
            processing = sum(
                1 for t in self.tasks.values() if t["status"] == "processing"
            )
            completed = sum(
                1 for t in self.tasks.values() if t["status"] == "completed"
            )
            failed = sum(
                1 for t in self.tasks.values() if t["status"] == "failed"
            )
        return {
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total": len(self.tasks),
        }

    # ------------------------------------------------------------------ #
    #  Work loop (ALL ranks)
    # ------------------------------------------------------------------ #

    def work_loop(self):
        """
        Synchronised work loop executed by every rank.

        Rank 0 polls the task queue and broadcasts a command:
            0 = idle (no work), 1 = generate, -1 = shutdown.
        All ranks follow the command in lockstep.
        """
        logger.info(f"Rank {self.rank} entering work loop.")

        while True:
            # Command tensor shared across ranks
            cmd_tensor = torch.zeros(
                1, dtype=torch.long, device=f"cuda:{self.local_rank}"
            )
            current_task_id = None

            if self.rank == 0:
                if self._shutdown:
                    cmd_tensor[0] = -1
                else:
                    try:
                        current_task_id = self.task_queue.get(timeout=0.5)
                        cmd_tensor[0] = 1
                    except queue.Empty:
                        cmd_tensor[0] = 0

            # Broadcast command to all ranks
            if self.world_size > 1:
                dist.broadcast(cmd_tensor, src=0)

            cmd = cmd_tensor.item()

            if cmd == -1:
                break
            elif cmd == 0:
                continue
            elif cmd == 1:
                self._process_task(current_task_id)

        logger.info(f"Rank {self.rank} exited work loop.")
        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()

    def _process_task(self, task_id: Optional[str]):
        """
        Run generation for one task across all ranks.

        Args:
            task_id: valid on rank 0 only; None on other ranks.
        """
        # Rank 0 broadcasts generation parameters
        if self.rank == 0:
            task = self.tasks[task_id]
            with self.lock:
                task["status"] = "processing"
            params = task["params"]

            # Resolve random seed on rank 0 so all ranks share the same noise.
            # Without this, seed=-1 causes each rank to independently pick a
            # different random seed inside model.generate(), leading to
            # inconsistent noise tensors and blurry / half-denoised output.
            seed = params.get("seed", -1)
            if seed < 0:
                seed = random.randint(0, sys.maxsize)
            params["seed"] = seed

            broadcast_data = [params]
            logger.info(
                f"Processing task {task_id}: "
                f"{params.get('prompt', '')[:80]}..."
            )
        else:
            broadcast_data = [None]

        if self.world_size > 1:
            dist.broadcast_object_list(broadcast_data, src=0)

        params = broadcast_data[0]

        try:
            # Load image
            img = Image.open(params["image_path"]).convert("RGB")

            # Resolve generation kwargs with config defaults
            frame_num = params.get("frame_num")
            if frame_num is None:
                frame_num = self.cfg.frame_num

            shift = params.get("shift")
            if shift is None:
                shift = self.cfg.sample_shift

            sampling_steps = params.get("sampling_steps")
            if sampling_steps is None:
                sampling_steps = self.cfg.sample_steps

            guide_scale = params.get("guide_scale")
            if guide_scale is None:
                guide_scale = self.cfg.sample_guide_scale
            elif isinstance(guide_scale, list):
                guide_scale = tuple(guide_scale)

            size_key = params.get("size", "480*832")
            max_area = MAX_AREA_CONFIGS.get(size_key, 480 * 832)

            gen_kwargs = {
                "input_prompt": params["prompt"],
                "img": img,
                "max_area": max_area,
                "frame_num": frame_num,
                "shift": shift,
                "sample_solver": params.get("sample_solver", "unipc"),
                "sampling_steps": sampling_steps,
                "guide_scale": guide_scale,
                "seed": params.get("seed", -1),
                "offload_model": False,
            }

            # Generate video (collective operation across all ranks)
            video = self.model.generate(**gen_kwargs)

            # Save result on rank 0
            if self.rank == 0:
                # Use client-specified save_path if provided, otherwise default
                client_save_path = params.get("save_path")
                if client_save_path:
                    save_path = client_save_path
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                else:
                    save_path = os.path.join(self.output_dir, f"{task_id}.mp4")
                save_video(
                    tensor=video[None],
                    save_file=save_path,
                    fps=self.cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                with self.lock:
                    task["status"] = "completed"
                    task["result_path"] = save_path
                    task["completed_at"] = datetime.now().isoformat()
                logger.info(f"Task {task_id} completed -> {save_path}")

                # Cleanup temp image if applicable
                if params.get("_temp_image"):
                    try:
                        os.remove(params["image_path"])
                    except OSError:
                        pass

            del video
            torch.cuda.synchronize()

        except Exception as e:
            logger.error(
                f"Task error on rank {self.rank}: {e}", exc_info=True
            )
            if self.rank == 0 and task_id:
                with self.lock:
                    task["status"] = "failed"
                    task["error"] = str(e)
                    task["completed_at"] = datetime.now().isoformat()

    def shutdown(self):
        """Signal all ranks to exit the work loop."""
        self._shutdown = True
