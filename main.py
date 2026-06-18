import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from api import app

if __name__ == "__main__":
    import uvicorn

    # Render sets PORT; must bind 0.0.0.0 so the platform can reach the app.
    port = int(os.environ["PORT"]) if "PORT" in os.environ else 8000
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"

    print(f"Starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
