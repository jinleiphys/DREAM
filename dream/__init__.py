"""Differentiable reduced emulation of parameterized linear systems."""

import jax

from . import _logging

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
