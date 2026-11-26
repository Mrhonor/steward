"""Observe a solid printed puzzle face and its cardboard back in MuJoCo."""

import json

import numpy as np
import pytest
from PIL import Image

from steward.assets.generators import building, dining_table, printed_board, steward_bot
from steward.authoring.service import png_b64, puzzle_box_m, puzzle_textures
from steward.ego.perception import decode_rgbd
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.world.sim import PerceptionConfig, WorldSim


ROBOT = "w:robot"
PIECES = {"target": 0.27, "reference-print": 0.09, "reference-back-near": -0.09, "reference-back-far": -0.27}


def _scene(target_front):
    room, robot, table = (factory().to_dict() for factory in (building, steward_bot, dining_table))
    for entity, eid in zip((room, robot, table), ("w:room", ROBOT, "w:table"), strict=True):
        entity["id"] = eid
    table["pose"]["value"]["p"] = [1.02, 0.0, 0.0]
    solid_blue = png_b64(Image.new("RGB", (128, 128), (44, 113, 181)), 128)
    _, dog = puzzle_textures("dog.jpg", puzzle_box_m(4), 4)
    pieces = []
    for (name, y), texture, front in zip(PIECES.items(), (solid_blue, dog[1][1], dog[1][2], dog[2][1]),
                                         (target_front, True, False, False), strict=True):
        piece = printed_board(width=0.115, depth=0.115, texture=texture, category="puzzle_piece").to_dict()
        piece["id"] = f"w:{name}"
        piece["pose"]["value"] = {"p": [0.92, y, 0.752 if front else 0.758],
                                     "q": [1.0, 0.0, 0.0, 0.0] if front else [0.0, 1.0, 0.0, 0.0]}
        pieces.append(piece)
    return {"mode": "world", "entities": [room, robot, table, *pieces]}, robot


class CameraLink(RobotLink):
    """Read the simulator's robot sensors synchronously on the rendering thread."""

    def __init__(self, sim):
        super().__init__("http://unused")
        self.sim = sim

    def sensors(self):
        return self.sim.sensors(ROBOT)

    def rgbd(self, camera=None):
        return self.sim.rgbd(ROBOT, camera)


@pytest.mark.parametrize("target_front", [True, False], ids=["front-up", "back-up"])
def test_plain_puzzle_face_can_be_distinguished_from_other_pieces(tmp_path, monkeypatch, target_front):
    monkeypatch.delenv("STEWARD_SEGMENTER_URL", raising=False)
    scene, robot = _scene(target_front)
    sim = WorldSim(scene, seed=3, perception=PerceptionConfig(pos_sigma_base=0, pos_sigma_per_m=0,
                                                            yaw_sigma=0, odom_pos_sigma=0, odom_yaw_sigma=0))
    link = CameraLink(sim)
    runtime = EgoRuntime(link, robot)
    try:
        for _ in range(4):
            sim.step_seconds(0.12)
            runtime._poll_once()
        tracks = sorted((track for track in runtime.estimator.tracks.values() if track.category == "puzzle_piece"),
                        key=lambda track: -track.pos[1])
        assert len(tracks) == len(PIECES)
        faces = {}
        samples = {}
        for name, track in zip(PIECES, tracks, strict=True):
            face = runtime.inspect(track.id, px=24)
            faces[name] = face
            interior = [cell for row in face["rows"][2:-2] for cell in row[2:-2] if cell is not None]
            assert interior, face["coverage"]
            samples[name] = np.median(interior, axis=0)
        backs = [samples["reference-back-near"], samples["reference-back-far"]]
        reference = np.mean(backs, axis=0)
        target_distance = float(np.linalg.norm(samples["target"] - reference))
        report = {name: {"entity": face["entity"], "plain": face["plain"], "interiorColors": face["interiorColors"],
                         "coverage": face["coverage"], "sourcePx": face["sourcePx"], "medianRgb": samples[name].tolist()}
                  for name, face in faces.items()}
        report["target"]["rgbDistanceFromReferenceBacks"] = round(target_distance, 2)
        (tmp_path / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (tmp_path / "ego-inspect.json").write_text(json.dumps(faces, separators=(",", ":")) + "\n", encoding="utf-8")
        (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for camera in ("head_camera", "wide_camera"):
            frame = decode_rgbd(link.rgbd(camera))
            Image.fromarray(frame["rgb"]).save(tmp_path / f"{camera}.png")
        print(json.dumps(report), tmp_path)

        assert all(face["coverage"] >= 0.9 and face["sourcePx"] >= 24 for face in faces.values())
        assert faces["target"]["plain"]
        recorded = runtime.fault_status()["observations"]
        assert [sample["entity"] for sample in recorded] == [track.id for track in tracks]
        assert recorded[0]["plain"] and recorded[0]["coverage"] >= 0.9
        assert all(faces[name]["plain"] for name in ("reference-back-near", "reference-back-far"))
        assert not faces["reference-print"]["plain"]
        assert np.linalg.norm(backs[0] - backs[1]) < 20
        if target_front:
            assert target_distance > 60
        else:
            assert target_distance < 20
    finally:
        runtime.stop()
        sim.stop()
