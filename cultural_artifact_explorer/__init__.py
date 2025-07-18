"""Top-level package for cultural_artifact_explorer tests and modules."""

# Allow relative imports when the package is not installed
from pathlib import Path
import sys

# Add the package's src directory to sys.path for local development
package_root = Path(__file__).resolve().parent
src_path = package_root / 'src'
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
