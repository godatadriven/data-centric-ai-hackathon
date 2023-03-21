from setuptools import find_packages, setup

setup(
    name="data-centric-hackathon",
    version="0.0.0",
    packages=find_packages(where="./src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "lightning",
        "jupyterlab",
        "loguru",
        "matplotlib",
        "ipywidgets",
        "pandas",
    ],
)
