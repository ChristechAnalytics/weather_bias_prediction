import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

HOST = "127.0.0.1"
PORT = 8000

if __name__ == "__main__":
    print(f"\n  API docs:  http://{HOST}:{PORT}/docs")
    print(f"  Health:    http://{HOST}:{PORT}/health\n")
    uvicorn.run("api:app", host=HOST, port=PORT)