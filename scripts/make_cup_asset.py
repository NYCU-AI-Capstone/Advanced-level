"""Procedurally generate a hollow, tapered cup as a self-contained USD asset.

Why this exists: the advanced-level task may not reuse the original cup USD.
Instead of pulling an external mesh (which dropped physics / was transparent),
we generate our own opaque, hollow, tapered cup from scratch. The mesh is a
surface of revolution (outer wall, base, inner floor, inner wall, rim) and the
USD carries its own physics (RigidBodyAPI + MassAPI + mesh CollisionAPI) so it
spawns as a valid Isaac Lab rigid body with no env_cfg changes.

Units are meters (metersPerUnit = 1.0). The origin is placed at the TOP RIM
center (rim at z=0, base at z=-H) to match the original cup's rim-origin, so it
is a drop-in for the existing CUP_Z placement.

Run on host (no GPU / pxr needed):
    python3 scripts/make_cup_asset.py
"""

from __future__ import annotations

import math
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "packages" / "simulator" / "assets"
KITCHEN_OBJECTS = ASSETS / "scenes" / "kitchen" / "objects"


def build_cup_mesh(
    r_top: float,
    r_bot: float,
    height: float,
    wall: float,
    floor: float,
    segments: int,
):
    """Return (points, face_counts, face_indices) for a hollow tapered cup.

    Origin at top-rim center; rim at z=0, base bottom at z=-height.
    """
    r_top_in = r_top - wall
    r_bot_in = r_bot - wall
    z_top = 0.0
    z_bot = -height
    z_floor = -height + floor

    points: list[tuple[float, float, float]] = []

    def ring(radius: float, z: float) -> list[int]:
        base = len(points)
        for k in range(segments):
            ang = 2.0 * math.pi * k / segments
            points.append((radius * math.cos(ang), radius * math.sin(ang), z))
        return list(range(base, base + segments))

    ring0 = ring(r_top, z_top)        # top outer rim
    ring1 = ring(r_bot, z_bot)        # bottom outer
    apex2 = len(points); points.append((0.0, 0.0, z_bot))       # base center (under)
    apex3 = len(points); points.append((0.0, 0.0, z_floor))     # inner floor center
    ring4 = ring(r_bot_in, z_floor)   # inner floor edge
    ring5 = ring(r_top_in, z_top)     # top inner rim

    counts: list[int] = []
    indices: list[int] = []

    def quad(a: int, b: int, c: int, d: int) -> None:
        counts.append(4)
        indices.extend((a, b, c, d))

    def tri(a: int, b: int, c: int) -> None:
        counts.append(3)
        indices.extend((a, b, c))

    n = segments
    for k in range(n):
        k1 = (k + 1) % n
        # outer wall (normal outward)
        quad(ring0[k], ring0[k1], ring1[k1], ring1[k])
        # bottom outer disk (normal down)
        tri(ring1[k], ring1[k1], apex2)
        # inner floor disk (normal up, into cup)
        tri(apex3, ring4[k], ring4[k1])
        # inner wall (normal inward)
        quad(ring4[k], ring4[k1], ring5[k1], ring5[k])
        # rim annulus (normal up)
        quad(ring5[k], ring5[k1], ring0[k1], ring0[k])

    return points, counts, indices, (r_top, height)


def _fmt_points(points) -> str:
    return ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in points)


def _fmt_ints(values) -> str:
    return ", ".join(str(v) for v in values)


def write_cup_usd(
    out_path: Path,
    color: tuple[float, float, float],
    mat_name: str,
    r_top: float = 0.040,
    r_bot: float = 0.032,
    height: float = 0.104,
    wall: float = 0.004,
    floor: float = 0.006,
    segments: int = 64,
    mass: float = 0.001,
) -> None:
    points, counts, indices, (rad, h) = build_cup_mesh(
        r_top, r_bot, height, wall, floor, segments
    )
    r, g, b = color
    usda = f"""#usda 1.0
(
    defaultPrim = "cup"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "cup" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {mass}

    def Mesh "geom" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "convexHull"
        uniform bool physics:collisionEnabled = 1
        uniform token subdivisionScheme = "none"
        bool doubleSided = 1
        float3[] extent = [(-{rad:.6f}, -{rad:.6f}, -{h:.6f}), ({rad:.6f}, {rad:.6f}, 0.0)]
        int[] faceVertexCounts = [{_fmt_ints(counts)}]
        int[] faceVertexIndices = [{_fmt_ints(indices)}]
        point3f[] points = [{_fmt_points(points)}]
        color3f[] primvars:displayColor = [({r}, {g}, {b})]
        rel material:binding = </cup/Looks/{mat_name}>
    }}

    def Scope "Looks"
    {{
        def Material "{mat_name}"
        {{
            token outputs:surface.connect = </cup/Looks/{mat_name}/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({r}, {g}, {b})
                float inputs:metallic = 0.0
                float inputs:roughness = 0.5
                float inputs:opacity = 1.0
                token outputs:surface
            }}
        }}
    }}
}}
"""
    out_path.write_text(usda)
    print(f"wrote {out_path}  ({len(points)} verts, {len(counts)} faces)")


def main() -> None:
    # Pink cup (used by shell_game for all cups, and by cup_stacking pink_cup)
    write_cup_usd(
        KITCHEN_OBJECTS / "PinkCup" / "PinkCup.usd",
        color=(0.92, 0.42, 0.58),
        mat_name="PinkCup_Mat",
    )
    # Blue cup (cup_stacking blue_cup) — slightly smaller, matching original specs
    write_cup_usd(
        KITCHEN_OBJECTS / "BlueCup" / "BlueCup.usd",
        color=(0.05, 0.16, 0.70),
        mat_name="BlueCup_Mat",
        r_top=0.037,
        r_bot=0.030,
        height=0.096,
    )


if __name__ == "__main__":
    main()
