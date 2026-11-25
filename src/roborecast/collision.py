import argparse, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.015)
    p.add_argument("--pool-gib", type=int, default=4)
    p.add_argument("--width", type=int, default=30)
    p.add_argument("--max-n-cutting", type=int, default=16)
    p.add_argument("--concave-edges", type=int, default=16)
    a = p.parse_args()
    import trimesh

    vendor = os.environ.get("CUACD_ROOT", os.environ.get("CUACD_VENDOR"))
    if vendor:
        candidate = Path(vendor)
        sys.path.insert(
            0, str(candidate.parent if candidate.name == "cuacd" else candidate)
        )
    import cuacd

    start = time.monotonic()
    mesh = trimesh.load(a.mesh, force="mesh", process=False)
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    center = (v.min(0) + v.max(0)) / 2
    scale = np.ptp(v, axis=0).max() / 2
    v = np.ascontiguousarray((v - center) / scale, dtype=np.float32)
    f = np.ascontiguousarray(f, dtype=np.int32)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with cuacd.Context(device=a.gpu, pool_bytes=a.pool_gib * 1024**3) as ctx:
        init = time.monotonic()
        audit = ctx.check_mesh(v, f)
        if audit["needs_remesh"]:
            v, f = ctx.preprocess(v, f, resolution=128)
        prep = time.monotonic()
        parts = ctx.lookahead_decompose(
            v,
            f,
            threshold=a.threshold,
            width=a.width,
            max_n_cutting=a.max_n_cutting,
            n_concave_edges=a.concave_edges,
            merge_hulls=True,
            decompose_components=True,
        )
        end = time.monotonic()
        rows = []
        for i, (_, _, vertices, faces) in enumerate(parts):
            vertices = vertices * scale + center
            if not np.isfinite(vertices).all() or not len(faces):
                raise ValueError("CuACD invalid part")
            path = out / f"hull_{i:03d}.obj"
            trimesh.Trimesh(vertices, faces, process=False).export(path)
            rows.append({"path": str(path.resolve())})
    if not rows:
        raise ValueError("CuACD produced no parts")
    report = {
        "parts": rows,
        "diagnostics": dict(
            backend="CuACD",
            version=importlib.metadata.version("cuacd"),
            license=importlib.metadata.metadata("cuacd").get(
                "License-Expression", "unknown"
            ),
            repository="https://github.com/eliphatfs/cuacd",
            source_sha256=hashlib.sha256(Path(a.mesh).read_bytes()).hexdigest(),
            generated_part_count=len(rows),
            elapsed_seconds=time.monotonic() - start,
            context_seconds=init - start,
            preprocess_seconds=prep - init,
            decompose_seconds=end - prep,
            audit=audit,
            threshold=a.threshold,
            gpu=a.gpu,
            pool_gib=a.pool_gib,
            width=a.width,
            max_n_cutting=a.max_n_cutting,
            n_concave_edges=a.concave_edges,
            normalization_scale=scale,
            source_module=cuacd.__file__,
            automatic_fallback=False,
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["diagnostics"]))

if __name__ == '__main__':
    main()
