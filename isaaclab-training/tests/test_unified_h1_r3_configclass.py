"""CPU regression for the real Isaac Lab configclass inheritance boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types


PRODUCT_ROOT = Path(__file__).parents[1]
ENV_SOURCE = PRODUCT_ROOT / "instinctlab/envs/unified_h1_direct_env.py"
CONFIGCLASS_SOURCE = Path(
    "/home/mint/rl_blackwell_test/repos/IsaacLab/source/isaaclab/isaaclab/utils/configclass.py"
)
H1_MESH_PATH = "/World/H1RayScene/CombinedMesh"


def _load_frozen_configclass():
    """Execute the frozen decorator while isolating unused dict helpers."""
    tree = ast.parse(CONFIGCLASS_SOURCE.read_text(), filename=str(CONFIGCLASS_SOURCE))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "dict"
        )
    ]
    module = types.ModuleType("unified_h1_r3_frozen_configclass")

    def forbidden(*args, **kwargs):
        raise AssertionError("dictionary conversion is outside this configclass probe")

    module.__dict__.update(
        class_to_dict=forbidden,
        update_class_from_dict=forbidden,
    )
    sys.modules[module.__name__] = module
    try:
        exec(compile(tree, str(CONFIGCLASS_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module.configclass


def test_h1_config_uses_instance_field_with_frozen_configclass_semantics():
    configclass = _load_frozen_configclass()

    @configclass
    class SensorCfg:
        prim_path: str = "/World/envs/env_.*/TASLAB/base_link"
        mesh_prim_paths: list[str] = ["/World/MID360StaticScene"]

    @configclass
    class BaseCfg:
        ray_caster: SensorCfg = SensorCfg()

    assert not hasattr(BaseCfg, "ray_caster")
    with __import__("pytest").raises(AttributeError):
        BaseCfg.ray_caster.replace(mesh_prim_paths=[H1_MESH_PATH])

    @configclass
    class H1Cfg(BaseCfg):
        ray_caster = BaseCfg().ray_caster.replace(mesh_prim_paths=[H1_MESH_PATH])

    base = BaseCfg()
    first = H1Cfg()
    second = H1Cfg()
    assert base.ray_caster.mesh_prim_paths == ["/World/MID360StaticScene"]
    assert first.ray_caster.prim_path == base.ray_caster.prim_path
    assert first.ray_caster.mesh_prim_paths == [H1_MESH_PATH]
    assert first.ray_caster is not second.ray_caster
    assert first.ray_caster.mesh_prim_paths is not second.ray_caster.mesh_prim_paths
    first.ray_caster.mesh_prim_paths.append("/mutation")
    assert second.ray_caster.mesh_prim_paths == [H1_MESH_PATH]


def test_real_h1_source_uses_fixed_instance_boundary_and_preserves_mesh_contract():
    configclass = _load_frozen_configclass()

    @configclass
    class SensorCfg:
        prim_path: str = "/World/envs/env_.*/TASLAB/base_link"
        mesh_prim_paths: list[str] = ["/World/MID360StaticScene"]

    @configclass
    class SceneCfg:
        num_envs: int = 2
        env_spacing: float = 4.0
        replicate_physics: bool = True

    @configclass
    class BaseCfg:
        state_space: int = 18
        scene: SceneCfg = SceneCfg()
        ray_caster: SensorCfg = SensorCfg()

    source = ENV_SOURCE.read_text()
    tree = ast.parse(source, filename=str(ENV_SOURCE))
    h1_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnifiedH1DirectEnvCfg"
    )
    ray_assignment = next(
        node
        for node in h1_config.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ray_caster" for target in node.targets))
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "ray_caster")
        )
    )
    expression = ast.unparse(ray_assignment.value)
    assert expression == "TaslabDirectEnvCfg().ray_caster.replace(mesh_prim_paths=[H1_RAY_MESH_PATH])"
    assert "mesh_prim_paths=[H1_RAY_MESH_PATH]" in source

    namespace = {
        "configclass": configclass,
        "TaslabDirectEnvCfg": BaseCfg,
        "InteractiveSceneCfg": SceneCfg,
        "CRITIC_H1_DIMENSION": 21,
        "H1_RAY_MESH_PATH": H1_MESH_PATH,
    }
    exec(
        compile(ast.Module(body=[h1_config], type_ignores=[]), str(ENV_SOURCE), "exec"),
        namespace,
    )
    first = namespace["UnifiedH1DirectEnvCfg"]()
    second = namespace["UnifiedH1DirectEnvCfg"]()
    assert first.state_space == 21
    assert first.scene.num_envs == 32
    assert first.scene.env_spacing == 100.0
    assert first.scene.replicate_physics is True
    assert first.ray_caster.prim_path == "/World/envs/env_.*/TASLAB/base_link"
    assert first.ray_caster.mesh_prim_paths == [H1_MESH_PATH]
    assert first.ray_caster is not second.ray_caster
    assert first.ray_caster.mesh_prim_paths is not second.ray_caster.mesh_prim_paths
