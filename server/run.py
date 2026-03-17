"""
Entry point for the Wan2.2 video generation server.

Launch with torchrun for multi-GPU support:

    torchrun --nproc_per_node=8 server/run.py \
        --ckpt_dir ./Wan2.2-I2V-A14B \
        --dit_fsdp --t5_fsdp --ulysses_size 8 \
        --port 8000

All 8 GPU processes load a shard of the model. Only rank 0 starts
the HTTP server; all ranks enter a synchronised work loop
to handle generation requests collectively.
"""

import argparse
import logging
import os
import signal
import sys
import threading

# Ensure project root is on sys.path so that 'import wan' works
# regardless of how torchrun sets up the Python path.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import uvicorn

from model_worker import ModelWorker
from app import create_app
import app as app_module

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan2.2 Video Generation Server"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        help="Model task type (default: i2v-A14B).",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Enable FSDP for T5.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Enable FSDP for DiT.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses sequence-parallel size (should equal nproc_per_node).",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Place T5 model on CPU.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server listen address.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server listen port.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Initialise model worker (all ranks) ----
    worker = ModelWorker(args)

    # ---- Start HTTP server on rank 0 only ----
    if worker.rank == 0:
        # Inject the worker into the app module
        app_module.worker = worker
        app = create_app()

        # Run uvicorn in a daemon thread so the main thread
        # can enter the work loop alongside other ranks.
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={
                "host": args.host,
                "port": args.port,
                "log_level": "info",
            },
            daemon=True,
        )
        server_thread.start()
        logger.info(f"HTTP server started on {args.host}:{args.port}")

    # ---- Handle graceful shutdown ----
    def _signal_handler(signum, frame):
        if worker.rank == 0:
            logger.info("Shutdown signal received.")
            worker.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ---- Enter work loop (all ranks) ----
    worker.work_loop()


if __name__ == "__main__":
    main()
