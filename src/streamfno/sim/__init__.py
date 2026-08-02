"""Many-partition queue simulator.

Independent scale parameters: N (partitions) and B (buffer depth).  The
empirical lag-density measure and the boundary-flux series J_B(t) are the
first-class outputs; see config.py for units and scaling conventions.
"""

from .config import SimConfig
from .engine import simulate
from .results import SimResult

__all__ = ["SimConfig", "SimResult", "simulate"]
