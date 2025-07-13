from setuptools import setup, find_packages

# Read requirements.txt for dependencies, excluding comments and empty lines
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='cultural_artifact_explorer',
    version='0.1.0',
    author='AI Agent Jules',
    author_email='jules@example.com',
    description='A multimodal retrieval system for exploring cultural artifacts.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/example/cultural_artifact_explorer', # Replace with your repo URL
    project_urls={
        'Bug Tracker': 'https://github.com/example/cultural_artifact_explorer/issues', # Replace
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Update if different
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha', # Or "4 - Beta", "5 - Production/Stable"
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Text Processing :: Linguistic',
    ],
    package_dir={'': 'src'},  # Tell setuptools that packages are under src
    packages=find_packages(where='src'),  # Find all packages in src
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'black',
            'isort',
            'mypy', # Optional for static typing
        ],
        # Example for GPU specific dependencies if not handled by main requirements
        # 'gpu': [
        #     'tensorflow-gpu', # Or specific CUDA version of PyTorch
        #     'faiss-gpu',
        # ]
    },
    entry_points={
        'console_scripts': [
            # Example: 'cae-run-pipeline=src.scripts.run_pipeline:main', # If scripts are structured as modules
            'cae-streamlit=scripts.launch_app:main_cli', # Example if launch_app.sh calls a python main
        ],
    },
    include_package_data=True, # To include non-code files specified in MANIFEST.in (if used)
    # If you have data files that should be included with the package (outside of src),
    # you might need to specify them, e.g., using package_data or MANIFEST.in.
    # For this project, 'configs' and 'data' are typically not part of the installed package itself,
    # but rather used during development/execution from the repository structure.
    # If 'configs' were to be part of the package:
    # package_data={
    #     'cultural_artifact_explorer': ['../configs/*.yaml'],
    # },
)

# Note:
# The `pyproject.toml` is the more modern way to specify build system requirements and project metadata.
# `setup.py` is still used by setuptools as the build script.
# For many projects, `setup.cfg` could also be used to hold static configuration instead of `setup.py` arguments.
# This setup.py assumes that your main source code is in a `src/` directory.
# If you have CLI scripts in the `scripts/` directory that you want to install,
# they should be Python modules with a main() function and listed in `entry_points`.
# Shell scripts like `launch_app.sh` are typically not handled by setup.py directly.
