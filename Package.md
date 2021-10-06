# Special files  for packaging

*"This tutorial walks you through how to package a simple Python project. It will show you how to add the necessary
files and structure to create the package, how to build the package, and how to upload it to the Python Package Index."*

from `https://packaging.python.org/tutorials/packaging-projects/`.


- `pyproject.toml`

This file tells to build tools (like pip and build) what is required to build the project.

- `setup.py`

This is the build script for setuptools. 
It tells setuptools about the package (such as the name and version) as well as
which code files to include.

- `LICENSE`
This tells users who install the package the terms under which they can use your package.