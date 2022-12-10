from setuptools import find_packages
from setuptools import setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    author="Wufei Ma, Artur Jesslen",
    author_email="wufeim@gmail.com",
    name="nemo",
    version="1.0.0",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Neural mesh models for 3D reasoning.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    packages=find_packages(include=["nemo", "nemo.*"]),
    url="https://github.com/wufeim/NeMo",
    zip_safe=False,
)
