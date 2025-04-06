import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEBUG = bool(os.getenv("DEBUG"))
DEBUG_DIR = Path(__file__).parent.parent.parent / "debug_images"
if DEBUG:
    DEBUG_DIR.mkdir(exist_ok=True)

LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
_SIMPLE_LOG_FORMAT = "%(asctime)s %(levelname)s  %(message)s"
_DEBUG_LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d  %(message)s"
LOG_FORMAT = _DEBUG_LOG_FORMAT if DEBUG else _SIMPLE_LOG_FORMAT
