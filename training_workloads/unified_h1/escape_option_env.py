"""Apply the frozen option to motor requests while leaving PPO samples intact."""

from instinctlab.control.escape_option import EscapeOption, METHOD_VERSION
from instinctlab.control.escape_recovery import EscapeRecovery, MIXED_VERSION, RECOVERY_VERSION
from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv


class EscapeOptionEnv(UnifiedH1DirectEnv):
    escape_option = None

    def set_escape_option(self, enabled, *, version=METHOD_VERSION):
        self.escape_option_version = version if enabled else None
        kwargs = dict(lanes=self.num_envs, device=self.device, dtype=self._actions.dtype)
        if not enabled:
            self.escape_option = None
        elif version == METHOD_VERSION:
            self.escape_option = EscapeOption(**kwargs)
        elif version in (MIXED_VERSION, RECOVERY_VERSION):
            self.escape_option = EscapeRecovery(**kwargs, recenter=version == RECOVERY_VERSION)
        else:
            raise ValueError('ESCAPE_OPTION_VERSION')

    def _pre_physics_step(self, actions):
        if isinstance(self.escape_option, EscapeRecovery):
            actions = self.escape_option.apply_recovery(self._policy, actions, self._control_pipelines[0].config)
        elif self.escape_option is not None:
            actions = self.escape_option.apply(self._policy, actions)
        super()._pre_physics_step(actions)

    def _get_rewards(self):
        reward = super()._get_rewards()
        if self.escape_option is not None:
            self.extras['pre_reset']['escape_option'] = {
                'active': self.escape_option.applied.detach().clone(),
                'direction_b': self.escape_option.direction.detach().clone(),
                'applied_action': self._actions.detach().clone(),
            }
            if isinstance(self.escape_option, EscapeRecovery):
                self.extras['pre_reset']['escape_option']['recovery'] = self.escape_option.recovering.detach().clone()
        return reward

    def _reset_idx(self, env_ids):
        if self.escape_option is not None:
            self.escape_option.reset(self._validate_reset_indices(env_ids))
        super()._reset_idx(env_ids)


class EscapeHoldoutEnv(EscapeOptionEnv):
    def __init__(self, cfg, *, layout_seed):
        self.layout_seed = layout_seed
        super().__init__(cfg)

    def _setup_scene(self):
        from .escape_holdout_scene import scene_geometry

        super()._setup_scene()
        self._replace_mesh(scene_geometry(self.scene.env_origins.detach().cpu(), self.layout_seed))

    def _replace_mesh(self, geometry):
        from instinctlab.envs.unified_h1_direct_env import _define_triangle_mesh
        from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH

        self.holdout_geometry = geometry
        points, faces, walls, wall_faces = geometry
        _define_triangle_mesh(H1_RAY_MESH_PATH, points, faces, collision=False)
        _define_triangle_mesh('/World/H1CollisionWalls/Walls', walls, wall_faces, collision=True)

    def holdout_preflight(self, output, *, scene='room-door-trunks-street-tube'):
        from training_workloads.scale_003.persistence import atomic_json_replace
        from .preflight import (initial_source, live_geometry_readback, persist_initial_source,
                                validate_initial_observability)

        geometry = live_geometry_readback(self)
        atomic_json_replace(output, 'initial_preflight/geometry.json', geometry)
        source = initial_source(self)
        persist_initial_source(output, source)
        observability = validate_initial_observability(source, output, full_rank=False)
        atomic_json_replace(output, 'initial_preflight.json', {
            'schema_version': 'unified-h1-preflight.v1', 'status': 'PASS',
            'scene': scene, 'num_envs': 32,
            'geometry': geometry, 'observability': observability,
        })


class EscapeRecoveryTrainingEnv(EscapeHoldoutEnv):
    from .recovery_training_scene import TUBE_LANES as tube_training_lanes

    def __init__(self, cfg):
        super().__init__(cfg, layout_seed=None)

    def _setup_scene(self):
        from .recovery_training_scene import geometry

        EscapeOptionEnv._setup_scene(self)
        self._replace_mesh(geometry(self.scene.env_origins.detach().cpu()))
