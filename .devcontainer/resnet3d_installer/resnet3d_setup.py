try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

setup(
    name="resnet3d",
    packages=find_packages(),
)
