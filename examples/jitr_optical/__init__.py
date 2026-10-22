"""Neutral-particle optical-potential example with a writable JIT cache."""

import os
from pathlib import Path
import tempfile

# External solver JIT caches belong outside both the package and source tree.
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'dream-jitr-numba'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dream-matplotlib'))
