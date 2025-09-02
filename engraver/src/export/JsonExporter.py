import json
from dataclasses import dataclass
from geometry.GeometryInt import GeometryInt
from geometry.GeoUtil import GeoUtil


@dataclass
class JsonExporter:

    @staticmethod
    def export(geom: GeometryInt, path: str) -> None:
        """Export as JSON: { "scale": int, "points": [ [[x,y], ...], ... ] }"""
        obj = {
            "scale": geom.scale,
            "points": [[[p.x, p.y] for p in pl.pts] for pl in geom.polylines],
        }

        data = json.dumps(obj, ensure_ascii=False,  indent=4, separators=(",", ":"))
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
