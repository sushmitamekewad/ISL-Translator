import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

existing_pythonpath = os.environ.get("PYTHONPATH", "")
if PROJECT_ROOT not in existing_pythonpath:
    os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else PROJECT_ROOT

from src.main import main

if __name__ == "__main__":
    main()
