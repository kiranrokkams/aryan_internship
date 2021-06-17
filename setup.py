import os
import os.path as op
from setuptools import PEP420PackageFinder
from distutils.core import setup

ROOT = op.dirname(op.abspath(__file__))

def get_version_info():
    """ Extract version information as a dictionary from version.py. """
    version_info = {}
    version_filename = os.path.join("tigerml", "version.py")
    with open(version_filename, "r") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
    exec(version_code, version_info)
    return version_info


setup(
    name="tigerml",
    version=get_version_info()["version"],
    package_dir={"": "src"},
    description="DS Templates",
    author="Tiger Analytics",
    packages=PEP420PackageFinder.find(where=str(ROOT))
)
