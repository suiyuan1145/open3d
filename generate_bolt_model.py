"""Generate printable bolt and matching nut meshes as OBJ and STL.

Default size:
- thread count: 20
- threaded length: 20 mm, equivalent to 2 cm
- outer diameter: 5 mm, equivalent to 0.5 cm

The generated files can be opened in Blender.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]


def add_vertex(vertices: list[tuple[float, float, float]], point: tuple[float, float, float]) -> int:
    vertices.append(point)
    return len(vertices) - 1


def add_quad(faces: list[tuple[int, int, int]], a: int, b: int, c: int, d: int) -> None:
    faces.append((a, b, c))
    faces.append((a, c, d))


def thread_radius(theta: float, z: float, pitch: float, core_radius: float, thread_height: float) -> float:
    phase = (z / pitch - theta / (2.0 * math.pi)) % 1.0
    distance = min(phase, 1.0 - phase)
    triangular_profile = max(0.0, 1.0 - 2.0 * distance)
    return core_radius + thread_height * triangular_profile


def create_threaded_shaft(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    length_mm: float,
    outer_radius_mm: float,
    thread_count: int,
    radial_segments: int,
    axial_segments: int,
) -> None:
    pitch = length_mm / thread_count
    thread_height = outer_radius_mm * 0.18
    core_radius = outer_radius_mm - thread_height

    grid: list[list[int]] = []
    for iz in range(axial_segments + 1):
        z = length_mm * iz / axial_segments
        row: list[int] = []
        for it in range(radial_segments):
            theta = 2.0 * math.pi * it / radial_segments
            radius = thread_radius(theta, z, pitch, core_radius, thread_height)
            row.append(add_vertex(vertices, (radius * math.cos(theta), radius * math.sin(theta), z)))
        grid.append(row)

    for iz in range(axial_segments):
        for it in range(radial_segments):
            a = grid[iz][it]
            b = grid[iz][(it + 1) % radial_segments]
            c = grid[iz + 1][(it + 1) % radial_segments]
            d = grid[iz + 1][it]
            add_quad(faces, a, b, c, d)

    bottom_center = add_vertex(vertices, (0.0, 0.0, 0.0))
    top_center = add_vertex(vertices, (0.0, 0.0, length_mm))
    for it in range(radial_segments):
        faces.append((bottom_center, grid[0][(it + 1) % radial_segments], grid[0][it]))
        faces.append((top_center, grid[-1][it], grid[-1][(it + 1) % radial_segments]))


def create_hex_head(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    head_radius_mm: float,
    head_height_mm: float,
) -> None:
    bottom_z = -head_height_mm
    top_z = 0.0
    bottom: list[int] = []
    top: list[int] = []
    for i in range(6):
        theta = math.pi / 6.0 + 2.0 * math.pi * i / 6.0
        x = head_radius_mm * math.cos(theta)
        y = head_radius_mm * math.sin(theta)
        bottom.append(add_vertex(vertices, (x, y, bottom_z)))
        top.append(add_vertex(vertices, (x, y, top_z)))

    for i in range(6):
        add_quad(faces, bottom[i], bottom[(i + 1) % 6], top[(i + 1) % 6], top[i])

    bottom_center = add_vertex(vertices, (0.0, 0.0, bottom_z))
    top_center = add_vertex(vertices, (0.0, 0.0, top_z))
    for i in range(6):
        faces.append((bottom_center, bottom[i], bottom[(i + 1) % 6]))
        faces.append((top_center, top[(i + 1) % 6], top[i]))


def hex_radius_at_angle(theta: float, circumradius: float) -> float:
    sector_angle = (theta + math.pi / 6.0) % (math.pi / 3.0) - math.pi / 6.0
    return circumradius * math.cos(math.pi / 6.0) / math.cos(sector_angle)


def create_matching_nut_mesh(
    thread_count: int,
    bolt_length_cm: float,
    bolt_width_cm: float,
    nut_height_cm: float,
    clearance_mm: float,
    radial_segments: int,
    axial_segments_per_thread: int,
) -> Mesh:
    bolt_length_mm = bolt_length_cm * 10.0
    bolt_outer_radius_mm = bolt_width_cm * 10.0 / 2.0
    pitch = bolt_length_mm / thread_count
    nut_height_mm = nut_height_cm * 10.0
    nut_thread_count = max(1, int(round(nut_height_mm / pitch)))
    axial_segments = max(nut_thread_count * axial_segments_per_thread, nut_thread_count * 8)

    thread_height = bolt_outer_radius_mm * 0.18
    inner_core_radius = bolt_outer_radius_mm - thread_height + clearance_mm
    outer_hex_radius = bolt_outer_radius_mm * 1.85

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    outer_grid: list[list[int]] = []
    inner_grid: list[list[int]] = []
    for iz in range(axial_segments + 1):
        z = nut_height_mm * iz / axial_segments
        outer_row: list[int] = []
        inner_row: list[int] = []
        for it in range(radial_segments):
            theta = 2.0 * math.pi * it / radial_segments
            outer_radius = hex_radius_at_angle(theta, outer_hex_radius)
            inner_radius = thread_radius(theta, z, pitch, inner_core_radius, thread_height)
            outer_row.append(add_vertex(vertices, (outer_radius * math.cos(theta), outer_radius * math.sin(theta), z)))
            inner_row.append(add_vertex(vertices, (inner_radius * math.cos(theta), inner_radius * math.sin(theta), z)))
        outer_grid.append(outer_row)
        inner_grid.append(inner_row)

    for iz in range(axial_segments):
        for it in range(radial_segments):
            next_it = (it + 1) % radial_segments
            add_quad(faces, outer_grid[iz][it], outer_grid[iz][next_it], outer_grid[iz + 1][next_it], outer_grid[iz + 1][it])
            add_quad(faces, inner_grid[iz][next_it], inner_grid[iz][it], inner_grid[iz + 1][it], inner_grid[iz + 1][next_it])

    for it in range(radial_segments):
        next_it = (it + 1) % radial_segments
        add_quad(faces, outer_grid[0][next_it], outer_grid[0][it], inner_grid[0][it], inner_grid[0][next_it])
        add_quad(faces, outer_grid[-1][it], outer_grid[-1][next_it], inner_grid[-1][next_it], inner_grid[-1][it])

    return Mesh(vertices=vertices, faces=faces)


def create_bolt_mesh(
    thread_count: int,
    length_cm: float,
    width_cm: float,
    radial_segments: int,
    axial_segments_per_thread: int,
) -> Mesh:
    length_mm = length_cm * 10.0
    outer_radius_mm = width_cm * 10.0 / 2.0
    axial_segments = max(thread_count * axial_segments_per_thread, thread_count * 8)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    create_hex_head(
        vertices=vertices,
        faces=faces,
        head_radius_mm=outer_radius_mm * 1.65,
        head_height_mm=outer_radius_mm * 1.2,
    )
    create_threaded_shaft(
        vertices=vertices,
        faces=faces,
        length_mm=length_mm,
        outer_radius_mm=outer_radius_mm,
        thread_count=thread_count,
        radial_segments=radial_segments,
        axial_segments=axial_segments,
    )
    return Mesh(vertices=vertices, faces=faces)


def face_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def write_obj(path: Path, mesh: Mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Bolt mesh generated in millimeters\n")
        for x, y, z in mesh.vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in mesh.faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_stl(path: Path, mesh: Mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("solid bolt\n")
        for a, b, c in mesh.faces:
            normal = face_normal(vertices[a], vertices[b], vertices[c])
            handle.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            handle.write("    outer loop\n")
            for index in (a, b, c):
                x, y, z = vertices[index]
                handle.write(f"      vertex {x:.6e} {y:.6e} {z:.6e}\n")
            handle.write("    endloop\n")
            handle.write("  endfacet\n")
        handle.write("endsolid bolt\n")


def mesh_bounds_mm(mesh: Mesh) -> dict[str, list[float] | float]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    size = maximum - minimum
    return {
        "min_xyz_mm": [float(value) for value in minimum],
        "max_xyz_mm": [float(value) for value in maximum],
        "size_xyz_mm": [float(value) for value in size],
        "width_x_mm": float(size[0]),
        "width_y_mm": float(size[1]),
        "height_z_mm": float(size[2]),
    }


def scale_mesh(mesh: Mesh, factor: float) -> Mesh:
    return Mesh(
        vertices=[(x * factor, y * factor, z * factor) for x, y, z in mesh.vertices],
        faces=list(mesh.faces),
    )


def write_report(path: Path, args: argparse.Namespace, bolt_mesh: Mesh, nut_mesh: Mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "unit": "millimeter",
        "input_parameters": {
            "thread_count": args.thread_count,
            "shaft_length_cm": args.length_cm,
            "shaft_length_mm": args.length_cm * 10.0,
            "thread_outer_diameter_cm": args.width_cm,
            "thread_outer_diameter_mm": args.width_cm * 10.0,
            "nut_height_cm": args.nut_height_cm,
            "nut_height_mm": args.nut_height_cm * 10.0,
            "clearance_mm": args.clearance_mm,
        },
        "bolt_bounds": mesh_bounds_mm(bolt_mesh),
        "nut_bounds": mesh_bounds_mm(nut_mesh),
        "note": "OBJ/STL coordinates are written in millimeters. Blender may import unitless OBJ/STL as Blender units; set import scale or scene units to millimeters if needed.",
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a threaded bolt OBJ/STL mesh for Blender.")
    parser.add_argument("--thread-count", type=int, default=20, help="Number of screw thread turns.")
    parser.add_argument("--length-cm", type=float, default=2.0, help="Threaded shaft length in centimeters.")
    parser.add_argument("--width-cm", type=float, default=0.5, help="Outer thread diameter in centimeters.")
    parser.add_argument("--nut-height-cm", type=float, default=0.5, help="Matching nut height in centimeters.")
    parser.add_argument("--clearance-mm", type=float, default=0.15, help="Extra radial clearance for printable nut fit.")
    parser.add_argument("--radial-segments", type=int, default=160, help="Circular mesh resolution.")
    parser.add_argument("--axial-segments-per-thread", type=int, default=16, help="Resolution along each thread turn.")
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs") / "bolt_20_threads", help="Output folder.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.thread_count <= 0:
        raise ValueError("--thread-count must be positive.")
    if args.length_cm <= 0 or args.width_cm <= 0:
        raise ValueError("--length-cm and --width-cm must be positive.")
    if args.nut_height_cm <= 0 or args.clearance_mm < 0:
        raise ValueError("--nut-height-cm must be positive and --clearance-mm must be non-negative.")
    if args.radial_segments < 24:
        raise ValueError("--radial-segments should be at least 24.")
    if args.axial_segments_per_thread < 4:
        raise ValueError("--axial-segments-per-thread should be at least 4.")

    bolt_mesh = create_bolt_mesh(
        thread_count=args.thread_count,
        length_cm=args.length_cm,
        width_cm=args.width_cm,
        radial_segments=args.radial_segments,
        axial_segments_per_thread=args.axial_segments_per_thread,
    )
    nut_mesh = create_matching_nut_mesh(
        thread_count=args.thread_count,
        bolt_length_cm=args.length_cm,
        bolt_width_cm=args.width_cm,
        nut_height_cm=args.nut_height_cm,
        clearance_mm=args.clearance_mm,
        radial_segments=args.radial_segments,
        axial_segments_per_thread=args.axial_segments_per_thread,
    )

    bolt_obj_path = args.output_dir / "bolt_20_threads_2cm_x_0_5cm.obj"
    bolt_stl_path = args.output_dir / "bolt_20_threads_2cm_x_0_5cm.stl"
    nut_obj_path = args.output_dir / "matching_nut_for_bolt_20_threads.obj"
    nut_stl_path = args.output_dir / "matching_nut_for_bolt_20_threads.stl"
    bolt_blender_obj_path = args.output_dir / "bolt_20_threads_2cm_x_0_5cm_blender_meters.obj"
    bolt_blender_stl_path = args.output_dir / "bolt_20_threads_2cm_x_0_5cm_blender_meters.stl"
    nut_blender_obj_path = args.output_dir / "matching_nut_for_bolt_20_threads_blender_meters.obj"
    nut_blender_stl_path = args.output_dir / "matching_nut_for_bolt_20_threads_blender_meters.stl"
    report_path = args.output_dir / "bolt_and_nut_dimensions_mm.json"
    bolt_blender_mesh = scale_mesh(bolt_mesh, 0.001)
    nut_blender_mesh = scale_mesh(nut_mesh, 0.001)
    write_obj(bolt_obj_path, bolt_mesh)
    write_stl(bolt_stl_path, bolt_mesh)
    write_obj(nut_obj_path, nut_mesh)
    write_stl(nut_stl_path, nut_mesh)
    write_obj(bolt_blender_obj_path, bolt_blender_mesh)
    write_stl(bolt_blender_stl_path, bolt_blender_mesh)
    write_obj(nut_blender_obj_path, nut_blender_mesh)
    write_stl(nut_blender_stl_path, nut_blender_mesh)
    write_report(report_path, args, bolt_mesh, nut_mesh)

    print(f"wrote bolt OBJ: {bolt_obj_path}")
    print(f"wrote bolt STL: {bolt_stl_path}")
    print(f"wrote nut OBJ: {nut_obj_path}")
    print(f"wrote nut STL: {nut_stl_path}")
    print(f"wrote Blender-meter bolt OBJ: {bolt_blender_obj_path}")
    print(f"wrote Blender-meter bolt STL: {bolt_blender_stl_path}")
    print(f"wrote Blender-meter nut OBJ: {nut_blender_obj_path}")
    print(f"wrote Blender-meter nut STL: {nut_blender_stl_path}")
    print(f"wrote dimensions report: {report_path}")
    print(f"bolt vertices={len(bolt_mesh.vertices)} faces={len(bolt_mesh.faces)}")
    print(f"nut vertices={len(nut_mesh.vertices)} faces={len(nut_mesh.faces)}")
    print(f"bolt size xyz mm={mesh_bounds_mm(bolt_mesh)['size_xyz_mm']}")
    print(f"nut size xyz mm={mesh_bounds_mm(nut_mesh)['size_xyz_mm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
