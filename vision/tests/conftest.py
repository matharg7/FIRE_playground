import sys
import os

VISION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPARSIMONY_REPO = os.path.join(VISION_DIR, "sparsimony")

for path in [VISION_DIR, SPARSIMONY_REPO]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
