import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

# Render sets PORT; bind to 0.0.0.0 so the platform can route traffic.
HOST = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))

if __name__ == "__main__":
    docs_host = "127.0.0.1" if HOST == "127.0.0.1" else "localhost"
    print(f"\n  API docs:  http://{docs_host}:{PORT}/docs")
    print(f"  Health:    http://{docs_host}:{PORT}/health\n")
    uvicorn.run("main:app", host=HOST, port=PORT)
