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

    def _reset_idx(self, env_ids):
        if self.escape_option is not None:
            self.escape_option.reset(self._validate_reset_indices(env_ids))
        super()._reset_idx(env_ids)
