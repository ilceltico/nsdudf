from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="My MC",
    ext_modules=cythonize(["_marching_cubes_lewiner_cy_pseudosdf.pyx"], include_path=[np.get_include()], language="c++"),
)
