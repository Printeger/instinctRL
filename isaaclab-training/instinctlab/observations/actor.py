"""Pure-tensor actor observation and four-frame history contract."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch

AZIMUTH_SAMPLES = 360
ELEVATION_SAMPLES = 59
RAY_COUNT = AZIMUTH_SAMPLES * ELEVATION_SAMPLES
HISTORY_LENGTH = 4
LIDAR_FIELDS_PER_FRAME = 3
STATE_FIELDS_PER_FRAME = 13
M3_MIN_RANGE_M = 0.1
M3_MAX_RANGE_M = 40.0
DEFAULT_GRAVITY_NORM_TOLERANCE = 1.0e-5

APPROVED_INPUT_NAMES = (
    "ranges_m",
    "validity",
    "sensor_frame_time_s",
    "policy_time_s",
    "body_angular_velocity_rad_s",
    "gravity_direction_body",
    "commanded_body_velocity_m_s",
    "previous_issued_body_velocity_m_s",
)

# This tuple is mirrored verbatim as JSON in OBSERVATION_CONTRACT.md.
OUTPUT_SLICE_PROVENANCE = (
    {"output": "lidar_grid", "slice": "[:,0::3,:,:]", "source": "ranges_m", "history": "oldest_to_newest"},
    {"output": "lidar_grid", "slice": "[:,1::3,:,:]", "source": "validity.float", "history": "oldest_to_newest"},
    {"output": "lidar_grid", "slice": "[:,2::3,:,:]", "source": "validity.float*exp(-age_s/tau_s)", "history": "oldest_to_newest"},
    {"output": "state_vec", "slice": "[:,0:52:13]+[0:3]", "source": "body_angular_velocity_rad_s", "history": "oldest_to_newest"},
    {"output": "state_vec", "slice": "[:,0:52:13]+[3:6]", "source": "gravity_direction_body", "history": "oldest_to_newest"},
    {"output": "state_vec", "slice": "[:,0:52:13]+[6:9]", "source": "commanded_body_velocity_m_s", "history": "oldest_to_newest"},
    {"output": "state_vec", "slice": "[:,0:52:13]+[9:12]", "source": "previous_issued_body_velocity_m_s", "history": "oldest_to_newest"},
    {"output": "state_vec", "slice": "[:,12:52:13]", "source": "policy_time_s-sensor_frame_time_s", "history": "oldest_to_newest"},
)


class ActorObservationHistory:
    """Build the exact M4 actor observation from explicitly approved tensors."""

    def __init__(
        self,
        tau_s: float = 0.5,
        gravity_norm_tolerance: float = DEFAULT_GRAVITY_NORM_TOLERANCE,
    ) -> None:
        if isinstance(tau_s, bool) or not isinstance(tau_s, (int, float)):
            raise TypeError("tau_s must be a real scalar")
        if not math.isfinite(float(tau_s)) or float(tau_s) <= 0.0:
            raise ValueError("tau_s must be finite and positive")
        if isinstance(gravity_norm_tolerance, bool) or not isinstance(
            gravity_norm_tolerance, (int, float)
        ):
            raise TypeError("gravity_norm_tolerance must be a real scalar")
        if not math.isfinite(float(gravity_norm_tolerance)) or float(gravity_norm_tolerance) <= 0.0:
            raise ValueError("gravity_norm_tolerance must be finite and positive")
        self.tau_s = float(tau_s)
        self.gravity_norm_tolerance = float(gravity_norm_tolerance)
        self._lidar_history: torch.Tensor | None = None
        self._state_history: torch.Tensor | None = None
        self._last_sensor_time_s: torch.Tensor | None = None
        self._last_policy_time_s: torch.Tensor | None = None
        self._has_time: torch.Tensor | None = None

    def update(
        self,
        ranges_m: torch.Tensor,
        validity: torch.Tensor,
        sensor_frame_time_s: torch.Tensor,
        policy_time_s: torch.Tensor,
        body_angular_velocity_rad_s: torch.Tensor,
        gravity_direction_body: torch.Tensor,
        commanded_body_velocity_m_s: torch.Tensor,
        previous_issued_body_velocity_m_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Validate one frame, append it, and return detached actor-visible copies."""
        float_inputs = {
            "ranges_m": ranges_m,
            "sensor_frame_time_s": sensor_frame_time_s,
            "policy_time_s": policy_time_s,
            "body_angular_velocity_rad_s": body_angular_velocity_rad_s,
            "gravity_direction_body": gravity_direction_body,
            "commanded_body_velocity_m_s": commanded_body_velocity_m_s,
            "previous_issued_body_velocity_m_s": previous_issued_body_velocity_m_s,
        }
        for name, value in float_inputs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have floating-point dtype")
        if not isinstance(validity, torch.Tensor):
            raise TypeError("validity must be a torch.Tensor")
        if validity.dtype != torch.bool:
            raise TypeError("validity must have boolean dtype")

        expected_shapes = {
            "ranges_m": (RAY_COUNT,),
            "validity": (RAY_COUNT,),
            "sensor_frame_time_s": (1,),
            "policy_time_s": (1,),
            "body_angular_velocity_rad_s": (3,),
            "gravity_direction_body": (3,),
            "commanded_body_velocity_m_s": (3,),
            "previous_issued_body_velocity_m_s": (3,),
        }
        all_inputs: Mapping[str, torch.Tensor] = {**float_inputs, "validity": validity}
        for name, trailing_shape in expected_shapes.items():
            value = all_inputs[name]
            if value.ndim != 2 or tuple(value.shape[1:]) != trailing_shape:
                dimensions = ", ".join(str(dimension) for dimension in trailing_shape)
                raise ValueError(f"{name} must have shape (N, {dimensions})")

        batch_size = ranges_m.shape[0]
        if batch_size <= 0 or any(value.shape[0] != batch_size for value in all_inputs.values()):
            raise ValueError("all inputs must share one non-empty batch dimension")
        devices = {value.device for value in all_inputs.values()}
        if len(devices) != 1:
            raise ValueError("all inputs must share one device")
        float_dtypes = {value.dtype for value in float_inputs.values()}
        if len(float_dtypes) != 1:
            raise ValueError("all floating inputs must share one dtype")
        dtype = ranges_m.dtype
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("floating inputs must use torch.float32 or torch.float64")
        if any(not bool(torch.isfinite(value).all().item()) for value in float_inputs.values()):
            raise ValueError("all floating inputs must be finite")
        if bool(((ranges_m < M3_MIN_RANGE_M) | (ranges_m > M3_MAX_RANGE_M)).any().item()):
            raise ValueError("ranges_m must preserve the M3 closed [0.1, 40.0] metre domain")

        gravity_norm = torch.linalg.vector_norm(gravity_direction_body, dim=-1)
        if bool((torch.abs(gravity_norm - 1.0) > self.gravity_norm_tolerance).any().item()):
            raise ValueError("gravity_direction_body must be unit length within configured tolerance")
        age_s = policy_time_s - sensor_frame_time_s
        if bool((age_s < 0.0).any().item()):
            raise ValueError("policy_time_s must not precede sensor_frame_time_s")

        self._ensure_storage(batch_size, ranges_m.device, dtype)
        assert self._has_time is not None
        assert self._last_sensor_time_s is not None
        assert self._last_policy_time_s is not None
        active = self._has_time
        if bool((active & (sensor_frame_time_s < self._last_sensor_time_s)).any().item()):
            raise ValueError("sensor_frame_time_s regressed for an initialized environment")
        if bool((active & (policy_time_s < self._last_policy_time_s)).any().item()):
            raise ValueError("policy_time_s regressed for an initialized environment")

        validity_float = validity.to(dtype=dtype)
        reliability = validity_float * torch.exp(-age_s / self.tau_s)
        reliability = reliability.clamp(0.0, 1.0)
        lidar_frame = torch.stack(
            (
                ranges_m.reshape(batch_size, ELEVATION_SAMPLES, AZIMUTH_SAMPLES),
                validity_float.reshape(batch_size, ELEVATION_SAMPLES, AZIMUTH_SAMPLES),
                reliability.reshape(batch_size, ELEVATION_SAMPLES, AZIMUTH_SAMPLES),
            ),
            dim=1,
        )
        state_frame = torch.cat(
            (
                body_angular_velocity_rad_s,
                gravity_direction_body,
                commanded_body_velocity_m_s,
                previous_issued_body_velocity_m_s,
                age_s,
            ),
            dim=-1,
        )

        assert self._lidar_history is not None
        assert self._state_history is not None
        self._lidar_history[:, :-1].copy_(self._lidar_history[:, 1:].clone())
        self._lidar_history[:, -1].copy_(lidar_frame)
        self._state_history[:, :-1].copy_(self._state_history[:, 1:].clone())
        self._state_history[:, -1].copy_(state_frame)
        self._last_sensor_time_s.copy_(sensor_frame_time_s)
        self._last_policy_time_s.copy_(policy_time_s)
        self._has_time.fill_(True)
        return self._actor_output_copy()

    def reset(self, env_indices: torch.Tensor | None = None) -> None:
        """Clear all environments, or only explicit environment indices."""
        if self._lidar_history is None:
            if env_indices is not None:
                raise RuntimeError("cannot selectively reset before the first update")
            return
        if env_indices is None:
            self._lidar_history.zero_()
            self._state_history.zero_()
            assert self._last_sensor_time_s is not None
            assert self._last_policy_time_s is not None
            assert self._has_time is not None
            self._last_sensor_time_s.zero_()
            self._last_policy_time_s.zero_()
            self._has_time.zero_()
            return
        if not isinstance(env_indices, torch.Tensor):
            raise TypeError("env_indices must be a torch.Tensor")
        if env_indices.ndim != 1 or env_indices.dtype != torch.long:
            raise TypeError("env_indices must be a rank-one torch.long tensor")
        if env_indices.device != self._lidar_history.device:
            raise ValueError("env_indices must share the history device")
        if env_indices.numel() == 0:
            return
        if bool(((env_indices < 0) | (env_indices >= self._lidar_history.shape[0])).any().item()):
            raise IndexError("env_indices contains an out-of-range environment")
        if env_indices.unique().numel() != env_indices.numel():
            raise ValueError("env_indices must not contain duplicates")
        self._lidar_history[env_indices] = 0.0
        self._state_history[env_indices] = 0.0
        assert self._last_sensor_time_s is not None
        assert self._last_policy_time_s is not None
        assert self._has_time is not None
        self._last_sensor_time_s[env_indices] = 0.0
        self._last_policy_time_s[env_indices] = 0.0
        self._has_time[env_indices] = False

    def _ensure_storage(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._lidar_history is None:
            self._lidar_history = torch.zeros(
                batch_size,
                HISTORY_LENGTH,
                LIDAR_FIELDS_PER_FRAME,
                ELEVATION_SAMPLES,
                AZIMUTH_SAMPLES,
                device=device,
                dtype=dtype,
            )
            self._state_history = torch.zeros(
                batch_size, HISTORY_LENGTH, STATE_FIELDS_PER_FRAME, device=device, dtype=dtype
            )
            self._last_sensor_time_s = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            self._last_policy_time_s = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            self._has_time = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
            return
        if (
            self._lidar_history.shape[0] != batch_size
            or self._lidar_history.device != device
            or self._lidar_history.dtype != dtype
        ):
            raise ValueError("batch, device, and dtype are fixed after the first update")

    def _actor_output_copy(self) -> dict[str, torch.Tensor]:
        assert self._lidar_history is not None
        assert self._state_history is not None
        batch_size = self._lidar_history.shape[0]
        return {
            "lidar_grid": self._lidar_history.reshape(
                batch_size,
                HISTORY_LENGTH * LIDAR_FIELDS_PER_FRAME,
                ELEVATION_SAMPLES,
                AZIMUTH_SAMPLES,
            ).detach().clone(),
            "state_vec": self._state_history.reshape(
                batch_size, HISTORY_LENGTH * STATE_FIELDS_PER_FRAME
            ).detach().clone(),
        }


__all__ = [
    "APPROVED_INPUT_NAMES",
    "AZIMUTH_SAMPLES",
    "ActorObservationHistory",
    "ELEVATION_SAMPLES",
    "HISTORY_LENGTH",
    "OUTPUT_SLICE_PROVENANCE",
    "RAY_COUNT",
]
