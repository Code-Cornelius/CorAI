import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python_libraries",
    version="0.0.1",
    author="Niels Cariou-Kotlarek, Bianca T. Catea",
    author_email="niels.carioukotlarek@gmail.com, cateabianca@gmail.com",
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
    package_dir={"priv_lib_error": "priv_lib_error",
                 "priv_lib_estimator": "priv_lib_estimator",
                 "priv_lib_metaclass": "priv_lib_metaclass",
                 "priv_lib_ml": "priv_lib_ml",
                 "priv_lib_plot": "priv_lib_plot",
                 "priv_lib_util": "priv_lib_util"},
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)