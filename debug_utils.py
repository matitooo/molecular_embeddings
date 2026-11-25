"""Verbose diagnostics for long-running training and preprocessing runs."""
import faulthandler
import logging
import os
import sys
from pathlib import Path

_LOG_PATH = None
_FAULT_HANDLE = None


def configure_debug_logging(log_path="results/logs/pipeline-debug.log", stack_interval=300):
    """Log detailed progress to stdout and a persistent file; dump stuck thread stacks."""
    global _LOG_PATH, _FAULT_HANDLE
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = os.path.abspath(log_path)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s [pid=%(process)d] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger("pipeline.debug")
    logger.info("Verbose diagnostics enabled; log=%s; pid=%s; python=%s",
                _LOG_PATH, os.getpid(), sys.version.replace("\n", " "))
    logger.info("Working directory: %s", os.getcwd())
    _FAULT_HANDLE = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)
    faulthandler.dump_traceback_later(stack_interval, repeat=True, file=_FAULT_HANDLE)
    logger.info("Periodic all-thread traceback dumps enabled every %s seconds", stack_interval)


def stop_debug_logging():
    global _FAULT_HANDLE
    try:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
    finally:
        if _FAULT_HANDLE is not None:
            _FAULT_HANDLE.close()
            _FAULT_HANDLE = None
