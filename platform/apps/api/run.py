from __future__ import annotations

import os
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(API_ROOT))

from platform_api import create_app


if __name__ == "__main__":
    create_app().run(
        host=os.getenv("PLATFORM_HOST", "127.0.0.1"),
        port=int(os.getenv("PLATFORM_PORT", "18888")),
        debug=False,
        threaded=True,
    )
