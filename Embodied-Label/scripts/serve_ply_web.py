#!/usr/bin/env python3
"""Serve a lightweight web viewer for PLY files on a headless server."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PLY Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    .app { display: grid; grid-template-columns: 360px 1fr; height: 100%; }
    .panel { border-right: 1px solid #ddd; padding: 12px; overflow: auto; background: #fafafa; }
    .viewer-wrap { position: relative; background: #111; }
    #viewer { width: 100%; height: 100%; display: block; }
    .row { margin-bottom: 10px; }
    .label { font-size: 12px; color: #555; margin-bottom: 4px; }
    input, button, select { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
    button { background: #f1f1f1; cursor: pointer; }
    button:hover { background: #e7e7e7; }
    .list { margin-top: 10px; display: grid; gap: 6px; }
    .item { font-size: 12px; text-align: left; }
    .hint { font-size: 12px; color: #666; line-height: 1.45; }
    .status {
      position: absolute; left: 10px; top: 10px; color: #eee; background: rgba(0,0,0,0.55);
      padding: 6px 10px; border-radius: 6px; font-size: 12px;
    }
    .axis-legend {
      position: absolute; right: 10px; top: 10px; color: #eee; background: rgba(0,0,0,0.55);
      padding: 6px 10px; border-radius: 6px; font-size: 12px; line-height: 1.4;
    }
    .axis-legend .x { color: #ff6666; }
    .axis-legend .y { color: #66ff66; }
    .axis-legend .z { color: #66aaff; }
  </style>
</head>
<body>
  <div class="app">
    <div class="panel">
      <div class="row">
        <div class="label">Filter by path substring</div>
        <input id="filter" placeholder="e.g. 08_world or toykitchen" />
      </div>
      <div class="row">
        <button id="refreshBtn">Refresh PLY List</button>
      </div>
      <div class="row">
        <div class="label">View mode</div>
        <select id="viewMode">
          <option value="original">Original RGB</option>
          <option value="semantic">Instance semantic</option>
          <option value="bboxes">3D AABB boxes</option>
        </select>
      </div>
      <div class="row">
        <div class="label">Point size</div>
        <input id="pointSize" type="range" min="0.001" max="0.04" step="0.001" value="0.006" />
      </div>
      <div class="row">
        <button id="flipZBtn">Flip Z Up/Down</button>
      </div>
      <div class="row hint">
        Tip (SSH tunnel):<br />
        <code>ssh -N -L 8765:127.0.0.1:8765 user@server</code><br />
        Then open <code>http://127.0.0.1:8765</code> locally.
      </div>
      <div id="list" class="list"></div>
    </div>
    <div class="viewer-wrap">
      <canvas id="viewer"></canvas>
      <div id="status" class="status">Ready</div>
      <div class="axis-legend">
        Coord Axes:<br />
        <span class="x">X</span> / <span class="y">Y</span> / <span class="z">Z</span>
      </div>
    </div>
  </div>

  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
      "three/examples/jsm/": "https://unpkg.com/three@0.164.1/examples/jsm/"
    }
  }
  </script>
  <script type="module">
    import * as THREE from "three";
    import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";
    import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

    const canvas = document.getElementById("viewer");
    const statusEl = document.getElementById("status");
    const listEl = document.getElementById("list");
    const filterEl = document.getElementById("filter");
    const refreshBtn = document.getElementById("refreshBtn");
    const viewModeEl = document.getElementById("viewMode");
    const pointSizeEl = document.getElementById("pointSize");
    const flipZBtn = document.getElementById("flipZBtn");

    let currentPoints = null;
    let currentPath = null;
    let boxesGroup = null;
    let currentMeta = null;
    const loader = new PLYLoader();
    let flipZSign = 1;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    const grid = new THREE.GridHelper(2, 20, 0x555555, 0x333333);
    grid.rotation.x = Math.PI / 2.0;
    scene.add(grid);
    const axesHelper = new THREE.AxesHelper(0.3);
    scene.add(axesHelper);
    const originDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.01, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    scene.add(originDot);

    const camera = new THREE.PerspectiveCamera(60, 1, 0.001, 1000);
    camera.up.set(0, 0, 1);
    camera.position.set(1.2, 1.2, 1.2);

    const controls = new TrackballControls(camera, canvas);
    controls.rotateSpeed = 4.0;
    controls.zoomSpeed = 1.5;
    controls.panSpeed = 1.0;
    controls.dynamicDampingFactor = 0.2;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const light = new THREE.DirectionalLight(0xffffff, 0.5);
    light.position.set(1, 2, 3);
    scene.add(light);

    function clearPoints() {
      if (!currentPoints) return;
      scene.remove(currentPoints);
      currentPoints.geometry.dispose();
      currentPoints.material.dispose();
      currentPoints = null;
    }

    function clearBoxes() {
      if (!boxesGroup) return;
      scene.remove(boxesGroup);
      boxesGroup.traverse((obj) => {
        if (obj.material && obj.material.dispose) {
          obj.material.dispose();
        }
      });
      boxesGroup = null;
    }

    function updateViewModeOptions(meta) {
      const hasInstance = !!(meta && meta.has_instance_id);
      for (const opt of viewModeEl.options) {
        if (opt.value === "semantic" || opt.value === "bboxes") {
          opt.disabled = !hasInstance;
        }
      }
      if (!hasInstance && viewModeEl.value !== "original") {
        viewModeEl.value = "original";
      }
    }

    async function fetchMeta(path) {
      const res = await fetch("/api/ply/meta?path=" + encodeURIComponent(path));
      if (!res.ok) {
        throw new Error("meta API failed");
      }
      return await res.json();
    }

    function drawInstanceBoxes(boxes) {
      clearBoxes();
      if (!boxes || boxes.length === 0) return;
      boxesGroup = new THREE.Group();
      boxesGroup.scale.z = flipZSign;
      for (const box of boxes) {
        const min = box.min || [0, 0, 0];
        const max = box.max || [0, 0, 0];
        const c = box.color_rgb || [255, 255, 255];
        const color = new THREE.Color(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0);
        const box3 = new THREE.Box3(
          new THREE.Vector3(min[0], min[1], min[2]),
          new THREE.Vector3(max[0], max[1], max[2])
        );
        const helper = new THREE.Box3Helper(box3, color);
        boxesGroup.add(helper);
      }
      scene.add(boxesGroup);
    }

    function buildPointUrl(path, mode) {
      if (mode === "semantic") {
        return "/api/ply/semantic?path=" + encodeURIComponent(path);
      }
      return "/api/ply/raw?path=" + encodeURIComponent(path);
    }

    function fitCameraToObject(object) {
      const box = new THREE.Box3().setFromObject(object);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z, 1e-3);
      const distance = maxDim * 1.8;
      camera.position.copy(center.clone().add(new THREE.Vector3(distance, distance, distance)));
      controls.target.copy(center);
      const axisLen = Math.max(0.08, Math.min(1.5, maxDim * 0.35));
      axesHelper.scale.setScalar(axisLen);
      const dotScale = Math.max(0.004, Math.min(0.05, maxDim * 0.02));
      originDot.scale.set(dotScale, dotScale, dotScale);
      controls.update();
    }

    function buildMaterial(geometry) {
      const size = Number(pointSizeEl.value || "0.006");
      const hasColor = !!geometry.getAttribute("color");
      if (hasColor) {
        return new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true });
      }
      return new THREE.PointsMaterial({ size, color: 0x8fd3ff, sizeAttenuation: true });
    }

    function loadPointCloud(path, mode) {
      return new Promise((resolve, reject) => {
        const url = buildPointUrl(path, mode);
        loader.load(
          url,
          (geometry) => {
            clearPoints();
            geometry.computeBoundingSphere();
            const material = buildMaterial(geometry);
            currentPoints = new THREE.Points(geometry, material);
            currentPoints.scale.z = flipZSign;
            scene.add(currentPoints);
            fitCameraToObject(currentPoints);
            resolve(geometry.attributes.position.count);
          },
          undefined,
          (err) => reject(err)
        );
      });
    }

    async function renderCurrentPath() {
      if (!currentPath) return;
      const mode = viewModeEl.value;

      if (mode === "bboxes") {
        if (!currentMeta || !currentMeta.has_instance_id) {
          viewModeEl.value = "original";
          return renderCurrentPath();
        }
        setStatus("Loading point cloud + bboxes: " + currentPath);
        const pointCount = await loadPointCloud(currentPath, "original");
        const res = await fetch("/api/ply/bboxes?path=" + encodeURIComponent(currentPath));
        if (!res.ok) {
          throw new Error("bbox API failed");
        }
        const payload = await res.json();
        drawInstanceBoxes(payload.boxes || []);
        setStatus(
          `Loaded ${currentPath} (${pointCount} points + ${payload.count || 0} boxes, mode=bboxes)`
        );
        return;
      }

      clearBoxes();
      if (mode === "semantic" && (!currentMeta || !currentMeta.has_instance_id)) {
        viewModeEl.value = "original";
      }
      const finalMode = viewModeEl.value;
      setStatus("Loading: " + currentPath);
      const count = await loadPointCloud(currentPath, finalMode);
      setStatus(`Loaded ${currentPath} (${count} points, mode=${finalMode})`);
    }

    async function loadScene(path) {
      currentPath = path;
      setStatus("Inspecting PLY: " + path);
      currentMeta = await fetchMeta(path);
      updateViewModeOptions(currentMeta);
      await renderCurrentPath();
    }

    async function refreshList() {
      const q = encodeURIComponent(filterEl.value.trim());
      const res = await fetch("/api/ply/list?contains=" + q);
      if (!res.ok) {
        throw new Error("list API failed");
      }
      const data = await res.json();
      listEl.innerHTML = "";
      if (!data.files || data.files.length === 0) {
        setStatus("No PLY files found");
        return;
      }
      for (const relPath of data.files) {
        const btn = document.createElement("button");
        btn.className = "item";
        btn.textContent = relPath;
        btn.onclick = () => {
          loadScene(relPath).catch((err) => {
            console.error("Load scene failed:", err);
            setStatus("Load failed: " + err.message);
          });
        };
        listEl.appendChild(btn);
      }
      setStatus(`Found ${data.files.length} PLY files`);
      if (!currentPath && data.files.length > 0) {
        await loadScene(data.files[0]);
      }
    }

    pointSizeEl.addEventListener("input", () => {
      if (!currentPoints) return;
      currentPoints.material.size = Number(pointSizeEl.value || "0.006");
      currentPoints.material.needsUpdate = true;
    });
    flipZBtn.addEventListener("click", () => {
      flipZSign = -flipZSign;
      if (currentPoints) {
        currentPoints.scale.z = flipZSign;
        currentPoints.updateMatrix();
      }
      if (boxesGroup) {
        boxesGroup.scale.z = flipZSign;
        boxesGroup.updateMatrix();
      }
      setStatus(flipZSign > 0 ? "Z orientation: normal" : "Z orientation: flipped");
    });
    viewModeEl.addEventListener("change", () => {
      if (currentPath) {
        renderCurrentPath().catch((err) => {
          console.error("Render failed:", err);
          setStatus("Render failed: " + err.message);
        });
      }
    });

    refreshBtn.addEventListener("click", () => {
      refreshList().catch((e) => {
        console.error("Refresh failed:", e);
        setStatus("Refresh failed: " + e.message);
      });
    });
    filterEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        refreshList().catch((err) => {
          console.error("Refresh failed:", err);
          setStatus("Refresh failed: " + err.message);
        });
      }
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w || canvas.height !== h) {
        renderer.setSize(w, h, false);
        camera.aspect = w / Math.max(h, 1);
        camera.updateProjectionMatrix();
        controls.handleResize();
      }
      renderer.render(scene, camera);
    }

    animate();
    refreshList().catch((err) => {
      console.error("Refresh failed:", err);
      setStatus("Refresh failed. Check browser console (F12).");
    });
  </script>
</body>
</html>
"""


class ServerState:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()
        self.cache_dir = Path("/tmp/embodied_label_ply_web_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_safe(self, rel_path: str) -> Path:
        candidate = (self.root_dir / rel_path).resolve()
        if self.root_dir != candidate and self.root_dir not in candidate.parents:
            raise HTTPException(status_code=400, detail="path escapes root directory")
        if not candidate.exists():
            raise HTTPException(status_code=404, detail="file not found")
        if not candidate.is_file():
            raise HTTPException(status_code=400, detail="path is not a file")
        if candidate.suffix.lower() != ".ply":
            raise HTTPException(status_code=400, detail="only .ply files are supported")
        return candidate

    def list_ply(self, contains: str = "") -> list[str]:
        files = []
        token = contains.lower().strip()
        for path in sorted(self.root_dir.rglob("*.ply")):
            rel = str(path.relative_to(self.root_dir)).replace("\\", "/")
            if token and token not in rel.lower():
                continue
            files.append(rel)
        return files

    @staticmethod
    def _read_ascii_ply_header(path: Path) -> tuple[list[str], int, int]:
        prop_names: list[str] = []
        n_vertices = None
        header_lines = 0
        with open(path, "r", encoding="utf-8") as fp:
            first = fp.readline().strip()
            header_lines += 1
            if first != "ply":
                raise HTTPException(status_code=400, detail="invalid PLY format")

            fmt = fp.readline().strip()
            header_lines += 1
            if "ascii" not in fmt:
                raise HTTPException(status_code=400, detail="only ascii PLY is supported for semantic/bbox")

            while True:
                line = fp.readline()
                if line == "":
                    raise HTTPException(status_code=400, detail="invalid PLY header (EOF)")
                header_lines += 1
                s = line.strip()
                if s.startswith("element vertex"):
                    n_vertices = int(s.split()[-1])
                elif s.startswith("property"):
                    prop_names.append(s.split()[-1])
                elif s == "end_header":
                    break

        if n_vertices is None:
            raise HTTPException(status_code=400, detail="invalid PLY: missing vertex element")
        return prop_names, n_vertices, header_lines

    def _load_ascii_ply_columns(self, path: Path) -> dict[str, np.ndarray]:
        prop_names, n_vertices, header_lines = self._read_ascii_ply_header(path)
        if n_vertices == 0:
            return {name: np.zeros((0,), dtype=np.float32) for name in prop_names}

        data = np.loadtxt(path, skiprows=header_lines, max_rows=n_vertices)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(prop_names):
            raise HTTPException(
                status_code=400,
                detail=f"invalid PLY columns: got {data.shape[1]}, expect {len(prop_names)}",
            )
        out: dict[str, np.ndarray] = {}
        for i, name in enumerate(prop_names):
            out[name] = data[:, i]
        return out

    def inspect_ply(self, path: Path) -> dict[str, Any]:
        prop_names, n_vertices, _ = self._read_ascii_ply_header(path)
        props = set(prop_names)
        return {
            "path": str(path),
            "n_vertices": int(n_vertices),
            "properties": prop_names,
            "has_instance_id": "instance_id" in props,
            "has_class_id": "class_id" in props,
            "has_rgb": all(name in props for name in ("red", "green", "blue")),
        }

    @staticmethod
    def _instance_color_map(instance_ids: np.ndarray) -> dict[int, tuple[int, int, int]]:
        ids = np.unique(instance_ids.astype(np.int64))
        ids = ids[ids != 0]
        ids = np.sort(ids)
        n = int(ids.shape[0])
        mapping: dict[int, tuple[int, int, int]] = {}
        if n == 0:
            return mapping
        for i, inst in enumerate(ids.tolist()):
            hue = float(i) / float(n)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            mapping[int(inst)] = (int(r * 255), int(g * 255), int(b * 255))
        return mapping

    def _cache_key(self, path: Path, suffix: str) -> str:
        st = path.stat()
        raw = f"{path}|{st.st_mtime_ns}|{st.st_size}|{suffix}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def build_semantic_ply(self, path: Path) -> Path:
        key = self._cache_key(path, "semantic_instance")
        out_path = self.cache_dir / f"{key}.ply"
        if out_path.exists():
            return out_path

        cols = self._load_ascii_ply_columns(path)
        for req in ("x", "y", "z", "instance_id"):
            if req not in cols:
                raise HTTPException(status_code=400, detail=f"PLY missing field: {req}")

        x = cols["x"].astype(np.float32)
        y = cols["y"].astype(np.float32)
        z = cols["z"].astype(np.float32)
        instance_ids = cols["instance_id"].astype(np.int64)

        color_map = self._instance_color_map(instance_ids)
        n = x.shape[0]
        colors = np.full((n, 3), 77, dtype=np.uint8)
        if color_map:
            for inst_id, rgb in color_map.items():
                mask = instance_ids == inst_id
                colors[mask, 0] = rgb[0]
                colors[mask, 1] = rgb[1]
                colors[mask, 2] = rgb[2]

        arr = np.column_stack([x, y, z, colors[:, 0], colors[:, 1], colors[:, 2]])
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write("ply\n")
            fp.write("format ascii 1.0\n")
            fp.write(f"element vertex {n}\n")
            fp.write("property float x\n")
            fp.write("property float y\n")
            fp.write("property float z\n")
            fp.write("property uchar red\n")
            fp.write("property uchar green\n")
            fp.write("property uchar blue\n")
            fp.write("end_header\n")
            np.savetxt(fp, arr, fmt="%.6f %.6f %.6f %d %d %d")
        return out_path

    def build_instance_bboxes(self, path: Path) -> dict[str, Any]:
        key = self._cache_key(path, "instance_bboxes")
        out_path = self.cache_dir / f"{key}.json"
        if out_path.exists():
            return json.loads(out_path.read_text(encoding="utf-8"))

        cols = self._load_ascii_ply_columns(path)
        for req in ("x", "y", "z", "instance_id"):
            if req not in cols:
                raise HTTPException(status_code=400, detail=f"PLY missing field: {req}")

        points = np.stack(
            [
                cols["x"].astype(np.float32),
                cols["y"].astype(np.float32),
                cols["z"].astype(np.float32),
            ],
            axis=1,
        )
        instance_ids = cols["instance_id"].astype(np.int64)
        color_map = self._instance_color_map(instance_ids)
        unique_instances = np.unique(instance_ids)
        unique_instances = unique_instances[unique_instances != 0]
        unique_instances = np.sort(unique_instances)

        boxes: list[dict[str, Any]] = []
        for inst_id in unique_instances.tolist():
            mask = instance_ids == inst_id
            if not np.any(mask):
                continue
            pts = points[mask]
            min_xyz = np.min(pts, axis=0).astype(float).tolist()
            max_xyz = np.max(pts, axis=0).astype(float).tolist()
            color = color_map.get(int(inst_id), (255, 255, 255))
            boxes.append(
                {
                    "instance_id": int(inst_id),
                    "num_points": int(np.sum(mask)),
                    "min": min_xyz,
                    "max": max_xyz,
                    "color_rgb": [int(color[0]), int(color[1]), int(color[2])],
                }
            )

        payload = {"count": len(boxes), "boxes": boxes}
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload


def create_app(root_dir: Path) -> FastAPI:
    app = FastAPI(title="Embodied-Label PLY Viewer")
    state = ServerState(root_dir)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "root_dir": str(state.root_dir)})

    @app.get("/api/ply/list")
    def list_ply(contains: str = Query(default="")) -> JSONResponse:
        files = state.list_ply(contains=contains)
        return JSONResponse({"root_dir": str(state.root_dir), "count": len(files), "files": files})

    @app.get("/api/ply/raw")
    def raw_ply(path: str = Query(..., description="Path relative to root-dir")) -> FileResponse:
        abs_path = state.resolve_safe(path)
        return FileResponse(abs_path, media_type="application/octet-stream", filename=abs_path.name)

    @app.get("/api/ply/meta")
    def ply_meta(path: str = Query(..., description="Path relative to root-dir")) -> JSONResponse:
        abs_path = state.resolve_safe(path)
        payload = state.inspect_ply(abs_path)
        return JSONResponse(payload)

    @app.get("/api/ply/semantic")
    def semantic_ply(path: str = Query(..., description="Path relative to root-dir")) -> FileResponse:
        abs_path = state.resolve_safe(path)
        semantic_path = state.build_semantic_ply(abs_path)
        return FileResponse(semantic_path, media_type="application/octet-stream", filename=semantic_path.name)

    @app.get("/api/ply/bboxes")
    def ply_bboxes(path: str = Query(..., description="Path relative to root-dir")) -> JSONResponse:
        abs_path = state.resolve_safe(path)
        payload = state.build_instance_bboxes(abs_path)
        return JSONResponse(payload)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a web viewer for PLY files")
    parser.add_argument("--root-dir", default="./logs", help="Directory to scan for .ply files")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise RuntimeError(f"root-dir not found: {root_dir}")

    app = create_app(root_dir)
    print(f"[ply-web] root-dir: {root_dir}")
    print(f"[ply-web] open: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
