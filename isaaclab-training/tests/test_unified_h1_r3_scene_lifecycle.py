"""CPU regression for the frozen InteractiveScene origins lifecycle."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from instinctlab.task.scene_h1 import combined_scene_triangles


PRODUCT_ROOT = Path(__file__).parents[1]
ENV_SOURCE = PRODUCT_ROOT / "instinctlab/envs/unified_h1_direct_env.py"


def _actual_setup_scene():
    tree = ast.parse(ENV_SOURCE.read_text(), filename=str(ENV_SOURCE))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnifiedH1DirectEnv"
    )
    method = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_setup_scene"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENV_SOURCE), "exec"), namespace)
    return namespace["_setup_scene"], namespace


def test_actual_setup_scene_consumes_origins_only_after_clone_and_before_sensors():
    setup_scene, namespace = _actual_setup_scene()
    events = []
    translated = {}

    class Scene:
        def __init__(self):
            # Faithful frozen replicate_physics=True initial state.
            self.env_origins = None
            self.articulations = {}
            self.sensors = {}

        def clone_environments(self, *, copy_from_source):
            assert copy_from_source is False
            assert "taslab" in self.articulations
            assert self.env_origins is None
            events.append("clone")
            self.env_origins = torch.zeros(32, 3)
            self.env_origins[:, 0] = torch.arange(32) * 100

        def filter_collisions(self, *, global_prim_paths):
            events.append(("filter", tuple(global_prim_paths)))

    def spawn_ground(path, cfg):
        events.append(("ground", path))

    def articulation(cfg):
        events.append("robot")
        return object()

    def spawn_combined(origins):
        assert isinstance(origins, torch.Tensor) and origins.shape == (32, 3)
        events.append("combined")
        points, indices, identities = combined_scene_triangles(origins)
        translated.update(points=points, indices=indices, identities=identities)

    def ray_caster(cfg):
        assert "points" in translated
        events.append("ray")
        return object()

    def contact_sensor(cfg):
        assert "points" in translated
        events.append("contact")
        return object()

    namespace.update(
        spawn_ground_plane=spawn_ground,
        GroundPlaneCfg=lambda: object(),
        Articulation=articulation,
        _spawn_unified_h1_combined_scene=spawn_combined,
        RayCaster=ray_caster,
        ContactSensor=contact_sensor,
    )
    scene = Scene()
    subject = SimpleNamespace(
        scene=scene,
        cfg=SimpleNamespace(robot=object(), ray_caster=object(), contact_sensor=object()),
        device="cpu",
    )
    setup_scene(subject)

    assert events == [
        ("ground", "/World/ground"),
        "robot",
        "clone",
        "combined",
        "ray",
        "contact",
        ("filter", ("/World/ground", "/World/H1CollisionWalls")),
    ]
    assert scene.env_origins.shape == (32, 3)
    assert len(translated["points"]) == 32 * 3 * 4
    assert len(translated["indices"]) == 32 * 3 * 6
    assert translated["points"][12][0] == 100 - 42
    assert translated["points"][-4][0] == 3100 - 6
    assert translated["identities"].count("env31:y_wall") == 2
    assert scene.sensors.keys() == {"mid360", "contact"}
