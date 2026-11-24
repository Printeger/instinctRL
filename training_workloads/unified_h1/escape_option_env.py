"""Apply the frozen option to motor requests while leaving PPO samples intact."""

from instinctlab.control.escape_option import EscapeOption
from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv


class EscapeOptionEnv(UnifiedH1DirectEnv):
    escape_option = None

    def set_escape_option(self, enabled):
        self.escape_option = (EscapeOption(lanes=self.num_envs, device=self.device, dtype=self._actions.dtype)
                              if enabled else None)

    def _pre_physics_step(self, actions):
        if self.escape_option is not None:
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
        from instinctlab.envs.unified_h1_direct_env import _define_triangle_mesh
        from instinctlab.task.scene_h1 import H1_RAY_MESH_PATH
        from .escape_holdout_scene import scene_geometry

        super()._setup_scene()
        self.holdout_geometry = scene_geometry(self.scene.env_origins.detach().cpu(), self.layout_seed)
        points, faces, walls, wall_faces = self.holdout_geometry
        _define_triangle_mesh(H1_RAY_MESH_PATH, points, faces, collision=False)
        _define_triangle_mesh('/World/H1CollisionWalls/Walls', walls, wall_faces, collision=True)
