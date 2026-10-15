import sys
import os
import warnings

# Suppress PyTorch architecture compatibility warning for newer GPUs (e.g. RTX 50-series sm_120)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

# Ensure the project root directory is always on sys.path and PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set PYTHONPATH environment variable so reloader child processes have access to 'src'
existing_pythonpath = os.environ.get("PYTHONPATH", "")
if PROJECT_ROOT not in existing_pythonpath:
    os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else PROJECT_ROOT

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"

    print("=====================================================")
    print("  Starting Indian Sign Language (ISL) Translator API ")
    print(f"  Project Root: {PROJECT_ROOT}                        ")
    print(f"  Server URL: http://localhost:{port}               ")
    print(f"  Swagger Docs: http://localhost:{port}/docs         ")
    print("=====================================================")

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=debug,
        app_dir=PROJECT_ROOT
    )

if __name__ == "__main__":
    main()
