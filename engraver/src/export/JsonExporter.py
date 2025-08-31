import json
from dataclasses import dataclass
from geometry.GeometryInt import GeometryInt


@dataclass
class JsonExporter:

    @staticmethod
    def export(geom: GeometryInt, path: str) -> None:
        """Export as JSON: { "scale": int, "polylines": [ [[x,y], ...], ... ] }"""
        obj = {
            "scale": geom.scale,
            "polylines": [[[p.x, p.y] for p in pl.pts] for pl in geom.polylines],
        }
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        if path == "-" or path == "stdout":
            print(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
