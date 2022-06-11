from pathlib import Path

import setuptools

PACKAGE_NAME = "yolov5faceInference"
VERSION = "0.0.1"
AUTHOR = "Vladimir Iglovikov"
EMAIL = "iglovikov@gmail.com"
DESCRIPTION = "Wrapper for yolov5FaceInference"
GITHUB_URL = "https://github.com/ternaus/yolov5faceInference"

parent_dir = Path(__file__).parent.absolute()

with (parent_dir / "README.md").open() as f:
    long_description = f.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=GITHUB_URL,
    package_dir={"face_detection_pytorch": "."},
    package_data={"": ["*net.pt"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
