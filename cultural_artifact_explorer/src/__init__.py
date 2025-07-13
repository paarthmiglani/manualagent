# This file makes 'src' a Python package.
# It can also be used to define top-level imports for the library if this project
# is intended to be importable as `import cultural_artifact_explorer_src; cultural_artifact_explorer_src.ocr...`
# However, with `package_dir={'': 'src'}` and `packages=find_packages(where='src')` in setup.py,
# the modules inside src (like ocr, nlp) will be importable directly, e.g. `from .ocr import ...`
# if the project is installed or the `src` directory is added to PYTHONPATH.

# Example: Expose a high-level API function or class
# from .pipeline.artifact_processor import ArtifactProcessor

__version__ = "0.1.0"

print("cultural_artifact_explorer src package initialized.")
