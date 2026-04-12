from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .segment_3d import Segment3D
from .toolpath_stats import ToolpathStats


@dataclass(frozen=True)
class ToolpathDocument:
    path: Path
    segments: Sequence[Segment3D]
    stats: ToolpathStats
