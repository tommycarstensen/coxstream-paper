"""Build the vendored Cython Efron streaming kernel (coxstream._kernel).

The project metadata lives in pyproject.toml; this file only declares the
compiled extension. Build in place for development with:

    pip install -e .
"""
import platform

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# -O3 -ffast-math lets the compiler auto-vectorise the O(p^2) inner loops.
# -march=native is added only on Linux: macOS Pythons are universal2 and
# -march=native breaks the cross-arch build.
_flags = ["-O3", "-ffast-math"]
if platform.system() == "Linux":
    _flags.append("-march=native")

setup(
    ext_modules=cythonize(
        [
            Extension(
                "coxstream._kernel",
                sources=["src/coxstream/_kernel.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=_flags,
            )
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
