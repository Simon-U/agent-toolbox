import os
from setuptools import setup, find_packages


# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()


setup(
    name="agent_toolbox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    author="Simon Unterbusch",
    author_email="simonunterbusch@unterbusch.com",
    description="A toolbox to create agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={},
)
