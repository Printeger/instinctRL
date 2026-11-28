"""Opt-in PhysX dynamic sphere and its corresponding MID360 return."""

from hashlib import sha256
from random import Random

import torch


RADIUS_M = .34
MASS_KG = 250.
MAX_SPEED_M_S = .95
PARK_HEIGHT_M = 65.
DYNAMIC_MODES = ('approach', 'recede', 'cross_left', 'cross_right', 'hover',
                 'hover_wall', 'static_travel', 'static_hover')
SWEEP_LOWER_XY_M = (-6., -1.35)
SWEEP_LENGTH_XY_M = (13., 2.7)
SWEEP_SPEED_XY_M_S = (.37, .58)
SWEEP_HEIGHT_M = 2.
SWEEP_ACCELERATION_RAMP_S = 1.


def street_sweep_phase(seed, lane, start_x):
    digest = sha256(f'independent-street-sweep-v1/{seed}/{lane}'.encode()).digest()
    schedule = Random(int.from_bytes(digest[:8], 'little'))
    x_period, y_period = (2.*length for length in SWEEP_LENGTH_XY_M)
    x = schedule.uniform(0., x_period)
    while abs(SWEEP_LOWER_XY_M[0]+min(x, x_period-x)-start_x) < 2.:
        x = schedule.uniform(0., x_period)
    return [x, schedule.uniform(0., y_period)]


def street_sweep_pose(origins, age_s, phase_m, *, ramp_s=0.):
    """Reflect a center along two independent, authored horizontal axes."""
    length = origins.new_tensor(SWEEP_LENGTH_XY_M)
    speed = origins.new_tensor(SWEEP_SPEED_XY_M_S)
    if ramp_s:
        acceleration = speed/ramp_s
        ramp_distance = .5*speed*ramp_s
        leg_time = length/speed+ramp_s
        offset = torch.minimum(phase_m, 2.*length-phase_m)
        start_time = torch.where(offset < ramp_distance, (2.*offset/acceleration).sqrt(),
            torch.where(offset > length-ramp_distance, leg_time-(2.*(length-offset)/acceleration).sqrt(),
                        ramp_s+(offset-ramp_distance)/speed))
        start_time = torch.where(phase_m > length, 2.*leg_time-start_time, start_time)
        clock = (start_time+age_s[..., None]).remainder(2.*leg_time)
        elapsed = torch.minimum(clock, 2.*leg_time-clock)
        offset = torch.where(elapsed < ramp_s, .5*acceleration*elapsed.square(),
            torch.where(elapsed > leg_time-ramp_s, length-.5*acceleration*(leg_time-elapsed).square(),
                        ramp_distance+speed*(elapsed-ramp_s)))
    else:
        traveled = (phase_m+age_s[..., None]*speed).remainder(2.*length)
        offset = torch.minimum(traveled, 2.*length-traveled)
    xy = origins[..., :2]+origins.new_tensor(SWEEP_LOWER_XY_M)+offset
    z = origins[..., 2:]+SWEEP_HEIGHT_M*torch.ones_like(xy[..., :1])
    return torch.cat((xy, z), dim=-1)


def street_sweep_initial_velocity(phase_m, ramp_s):
    """Velocity at the authored spatial phase for a ramped reflection."""
    length = phase_m.new_tensor(SWEEP_LENGTH_XY_M)
    speed = phase_m.new_tensor(SWEEP_SPEED_XY_M_S)
    offset = torch.minimum(phase_m, 2.*length-phase_m)
    distance_to_turn = torch.minimum(offset, length-offset).clamp_min(0.)
    magnitude = (2.*speed/ramp_s*distance_to_turn).sqrt().minimum(speed)
    direction = torch.where(phase_m < length, magnitude, -magnitude)
    return torch.nn.functional.pad(direction, (0, 1))


def spawn(scene):
    from isaaclab import sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    cfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/R6MovingSphere',
        spawn=sim_utils.SphereCfg(
            radius=RADIUS_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True,
                                                        linear_damping=0., angular_damping=0.),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_KG)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., PARK_HEIGHT_M)))
    ball = RigidObject(cfg)
    scene.rigid_objects['r6_moving_sphere'] = ball
    return ball


def sphere_range(origin, directions, center, *, radius=RADIUS_M):
    """Ray-sphere first surface distance in metres with +inf for no hit."""
    relative = center - origin
    projection = torch.bmm(directions, relative[:, :, None]).squeeze(-1)
    discriminant = projection.square() - relative.square().sum(1, keepdim=True) + radius**2
    root = discriminant.clamp_min(0.).sqrt()
    near, far = projection - root, projection + root
    depth = torch.where(near > 0, near, far)
    return torch.where((discriminant >= 0) & (depth >= .1) & (depth <= 40.), depth, torch.inf)


def trajectories(origins, axis, modes):
    """Linear trajectories in a lane's local command/left basis."""
    lateral = torch.stack((-axis[:, 1], axis[:, 0], torch.zeros_like(axis[:, 0])), dim=-1)
    approaching = (modes == 0) | (modes == 4) | (modes == 5)
    crossing = (modes == 2) | (modes == 3)
    sign = torch.where(modes == 3, 1., -1.)[:, None]
    start = origins + axis * torch.where(approaching, 2.75, torch.where(modes == 1, 1.45, 1.25))[:, None]
    start += lateral * crossing[:, None] * sign * 2.2
    velocity = axis * torch.where(approaching, -.42, torch.where(modes == 1, .36, 0.))[:, None]
    velocity -= lateral * crossing[:, None] * sign * .58
    active = modes < 6
    return start, velocity, active


def travel_clock(age, half_period_s):
    if half_period_s is None:
        return age, torch.ones_like(age)
    phase = age.remainder(2 * half_period_s)
    returning = phase >= half_period_s
    return torch.where(returning, 2 * half_period_s - phase, phase), torch.where(returning, -1., 1.)


class MovingSphere:
    def __init__(self, asset, origins):
        self.asset = asset
        self.parked = origins.clone()
        self.parked[:, 2] += PARK_HEIGHT_M
        self.indices = torch.arange(len(origins), device=origins.device, dtype=torch.long)
        self.age = torch.zeros(len(origins), device=origins.device, dtype=torch.float32)
        self.motion_ticks = None
        self.end_travel_s = torch.full_like(self.age, torch.inf)
        self.servo_velocity = torch.zeros_like(origins)
        self.enabled = False
        self.half_period_s = None
        self.departure_delay_s = None
        self.sweep_phase_m = None
        self.sweep_dt_s = None
        self.pre_reset = None
        self.latest = None
        self.verified_order = False
        self.interval_start = None
        self.contact = torch.zeros(len(origins), 1, device=origins.device, dtype=torch.bool)
        self.contact_samples = 0

    def configure(self, sensor_origins, axis, modes, *, half_period_s=None, departure_delay_s=None,
                  exact_clock=False, sweep_phase_m=None, sweep_ramp_s=0.):
        if self.enabled:
            raise RuntimeError('Existing moving-sphere block is active')
        if sweep_phase_m is not None and (not exact_clock or sweep_phase_m.shape != sensor_origins[:, :2].shape
                or sweep_phase_m.device != sensor_origins.device or sweep_phase_m.dtype != sensor_origins.dtype
                or not bool(sweep_phase_m.isfinite().all()) or bool(((modes < 6) & ~((modes == 2) | (modes == 3))).any())):
            raise ValueError('MOVING_SPHERE_STREET_SWEEP_PROFILE')
        if sweep_phase_m is not None:
            limit = 2.*sweep_phase_m.new_tensor(SWEEP_LENGTH_XY_M)
            if bool(((sweep_phase_m < 0.) | (sweep_phase_m > limit)).any()):
                raise ValueError('MOVING_SPHERE_STREET_SWEEP_PHASE')
        if (type(sweep_ramp_s) not in (int, float) or sweep_ramp_s not in (0., SWEEP_ACCELERATION_RAMP_S)
                or (sweep_ramp_s and sweep_phase_m is None)):
            raise ValueError('MOVING_SPHERE_STREET_SWEEP_RAMP')
        if not self.verified_order:
            observed = self.asset.root_physx_view.get_transforms()[:, :3]
            if observed.shape != self.parked.shape:
                raise RuntimeError('Moving-sphere PhysX view lane count differs from Isaac scene origins')
            errors = (observed - self.parked).abs().amax(dim=1)
            if not bool(errors.amax() <= .003):
                lane = int(errors.argmax())
                raise RuntimeError(f'Moving-sphere lane {lane} parked at {observed[lane].tolist()}, '
                                   f'expected {self.parked[lane].tolist()}')
            masses = self.asset.root_physx_view.get_masses()
            if masses.numel() != len(self.parked) or not bool((masses - MASS_KG).abs().amax() <= .01):
                raise RuntimeError('Measured moving-sphere mass differs from the configured PhysX mass')
            self.verified_order = True
        self.start, self.velocity, self.active = trajectories(sensor_origins, axis, modes)
        self.start = torch.where(self.active[:, None], self.start, self.parked)
        # The wall-hover and receding lanes end before the H1 wall planes.
        self.end_travel_s = torch.where(modes == 5, 8.7, torch.where(modes == 1, 10., torch.inf))
        self.half_period_s = half_period_s
        self.departure_delay_s = departure_delay_s
        self.sweep_phase_m = sweep_phase_m
        self.sweep_ramp_s = sweep_ramp_s
        self.sweep_dt_s = None
        self.sweep_start_velocity = None
        if sweep_phase_m is not None:
            self.sweep_origins = self.parked.clone()
            self.sweep_origins[:, 2] -= PARK_HEIGHT_M
            self.start = self._street_sweep_pose(torch.zeros_like(self.age))
            if sweep_ramp_s:
                velocity = street_sweep_initial_velocity(sweep_phase_m, sweep_ramp_s)
                self.sweep_start_velocity = velocity*self.active[:, None]
        self.motion_ticks = torch.zeros_like(self.indices) if exact_clock else None
        self.enabled = True
        self.reset(self.indices)

    def _travel_clock(self):
        if self.departure_delay_s is None:
            return travel_clock(self.age, self.half_period_s)
        elapsed = self.age - self.departure_delay_s
        travel_time, direction = travel_clock(elapsed.clamp_min(0.), self.half_period_s)
        return travel_time, direction * (elapsed >= 0.)

    def _pose(self):
        if self.sweep_phase_m is not None:
            return self._street_sweep_pose(self.age)
        travel_time, _ = self._travel_clock()
        return self.start + travel_time.minimum(self.end_travel_s)[:, None] * self.velocity

    def _street_sweep_pose(self, age_s):
        pose = street_sweep_pose(self.sweep_origins, age_s, self.sweep_phase_m, ramp_s=self.sweep_ramp_s)
        return torch.where(self.active[:, None], pose, self.parked)

    def reset(self, indices):
        if self.enabled and len(indices):
            self.age[indices] = 0.
            if self.motion_ticks is not None:
                self.motion_ticks[indices] = 0
            self.servo_velocity[indices] = 0.
            start_velocity = None if self.sweep_start_velocity is None else self.sweep_start_velocity[indices]
            self._place(self._pose()[indices], indices, velocity_w=start_velocity)

    def _place(self, position, indices=None, *, velocity_w=None):
        quaternion = position.new_zeros((len(position), 4))
        quaternion[:, 0] = 1.
        self.asset.write_root_pose_to_sim(torch.cat((position, quaternion), dim=1), env_ids=indices)
        velocity = position.new_zeros((len(position), 6))
        if velocity_w is not None:
            velocity[:, :3] = velocity_w
        self.asset.write_root_velocity_to_sim(velocity, env_ids=indices)

    def push(self, dt):
        if not self.enabled:
            return
        if self.sweep_phase_m is not None:
            self.sweep_dt_s = dt
        if self.motion_ticks is None:
            self.age.add_(dt)
        else:
            self.motion_ticks.add_(1)
            self.age.copy_(self.motion_ticks * dt)
        center = self.asset.root_physx_view.get_transforms()[:, :3]
        required_velocity = (self._pose() - center) / dt
        scale = (MAX_SPEED_M_S / required_velocity.norm(dim=1, keepdim=True).clamp_min(.01)).clamp_max(1.)
        # The GPU PhysX backend integrates this velocity and computes contacts.
        # Pose writes occur only at resets and between blocks.
        self.servo_velocity.copy_(required_velocity * scale)
        self.asset.write_root_velocity_to_sim(torch.cat((self.servo_velocity, torch.zeros_like(self.servo_velocity)), dim=1))

    def begin_interval(self, sim_tick):
        self.interval_start = sim_tick
        self.contact_samples = 0
        self.contact.zero_()

    def sample_contact(self, contact_sensor, threshold, sim_tick):
        # Before each substep, sample the contact result of the previous substep.
        if self.interval_start is not None and sim_tick > self.interval_start:
            force = contact_sensor.data.net_forces_w
            self.contact |= (force.norm(dim=-1) > threshold).any(dim=1, keepdim=True)
            self.contact_samples += 1

    def merge(self, hits, origin, directions, *, capture, tick):
        center = self.asset.root_physx_view.get_transforms()[:, :3].clone()
        pose = self._pose()
        error = (center-pose).abs().amax()
        if not bool(error <= .003):
            raise RuntimeError(f'Moving sphere disagrees with PhysX readback at tick {tick}: {float(error):.6f} m')
        moving_ranges = sphere_range(origin, directions, center)
        static_ranges = torch.nan_to_num((hits-origin[:, None]).norm(dim=-1), nan=torch.inf, posinf=torch.inf)
        overwrite = self.active[:, None] & (moving_ranges < static_ranges)
        finite_range = torch.where(overwrite, moving_ranges, 0.)
        result = torch.where(overwrite[:, :, None], origin[:, None] + directions * finite_range[:, :, None], hits)
        witnesses = torch.where(overwrite, moving_ranges, 41.).argmin(dim=1)
        lane = self.indices
        velocity = self.asset.root_physx_view.get_velocities()[:, :3].clone()
        if self.sweep_phase_m is None:
            travel_time, direction = self._travel_clock()
            direction *= travel_time < self.end_travel_s
            expected_velocity = direction[:, None] * self.velocity
        elif self.sweep_dt_s is None:
            expected_velocity = torch.zeros_like(velocity)
        else:
            prior = self._street_sweep_pose((self.age-self.sweep_dt_s).clamp_min(0.))
            expected_velocity = (pose-prior)/self.sweep_dt_s
        if self.sweep_start_velocity is not None:
            expected_velocity = torch.where(self.age[:, None] > 0., expected_velocity, self.sweep_start_velocity)
        self.latest = dict(tick=tick, applied_age_s=self.age.clone(), active=self.active.clone(),
                           sphere_position_w=center, expected_position_w=pose, sphere_velocity_w=velocity,
                           expected_velocity_w=expected_velocity, servo_velocity_w=self.servo_velocity.clone(),
                           sensor_position_w=origin.clone(), winning_beams=overwrite.sum(1),
                           sphere_sensor_clearance_m=(center-origin).norm(dim=1)-RADIUS_M,
                           witness_beam=witnesses, witness_direction_w=directions[lane, witnesses],
                           witness_range_m=finite_range[lane, witnesses])
        if capture:
            self.pre_reset = self.latest
        return result

    def finish(self):
        if self.enabled:
            self.servo_velocity.zero_()
            self._place(self.parked)
            self.enabled = False
