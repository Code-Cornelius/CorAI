import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python_libraries",
    version="0.0.1",
    author="Niels Cariou-Kotlarek",
    author_email="cateabianca@gmail.com",
    description="A collection of python libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Code-Cornelius/python_libraries",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "priv_lib_error"},
    packages=setuptools.find_packages(where="priv_lib_error"),
    python_requires=">=3.6",
)