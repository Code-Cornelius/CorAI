import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="corai",
    version="1.303",
    author="Niels D. Kotlarek, Bianca T. Catea",
    author_email="niels.carioukotlarek@gmail.com",
    description="A collection of python libraries.",
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
    install_requires=[
        'setuptools>=56.0.0',
        'matplotlib>=3.3.1',
        'numpy>=1.21.0',
        'pandas>=1.2.3',
        'seaborn>=0.11.1',
        'torch>=1.9.0+cu102',
        'sklearn>=0.0',
        'scikit-learn>=0.23.2',
        'tqdm>=4.61.1',
        'Keras>=2.4.3',
        'Pillow>=8.2.0',
        'scipy>=1.7.0'
    ],
    package_dir={"corai_error": "corai_error",
                 "corai_estimator": "corai_estimator",
                 "corai_metaclass": "corai_metaclass",
                 "corai": "corai",
                 "corai_plot": "corai_plot",
                 "corai_util": "corai_util"},
    packages=setuptools.find_packages(),
    python_requires=">=3.7.12",
)