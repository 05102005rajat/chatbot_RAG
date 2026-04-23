import os
import sys

# Make project root importable so `import rag_engine` works from tests/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
