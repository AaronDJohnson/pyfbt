import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyfbt",
    version="0.0.1",
    author="Aaron Johnson",
    author_email="johnsoad@uwm.edu",
    description="Frequency based Teukolsky point source Python code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AaronDJohnson/pyfbt",
    packages=setuptools.find_packages(),
    install_requires=[
        'gmpy',
        'numpy',
        'mpmath',
        'matplotlib',
        'numba',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)