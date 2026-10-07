"""Build coxph_cython.pyx and coxph_cython_f32.pyx in-place.

Run from src/coxbench/:
    python setup_cython.py build_ext --inplace
"""
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

_COMPILER_FLAGS = ["-O3", "-ffast-math"]
_DIRECTIVES = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
}

exts = [
    Extension(
        "coxph_cython",
        sources=["coxph_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=_COMPILER_FLAGS,
    ),
    Extension(
        "coxph_cython_f32",
        sources=["coxph_cython_f32.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=_COMPILER_FLAGS,
    ),
]

setup(
    name="coxph_cython",
    ext_modules=cythonize(exts, compiler_directives=_DIRECTIVES),
)
