"""Opt-in PhysX dynamic sphere and its corresponding MID360 return."""

import torch


RADIUS_M = .34
MASS_KG = 250.
MAX_SPEED_M_S = .95
PARK_HEIGHT_M = 65.
DYNAMIC_MODES = ('approach', 'recede', 'cross_left', 'cross_right', 'hover',
                 'hover_wall', 'static_travel', 'static_hover')


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
        self.pre_reset = None
        self.latest = None
        self.verified_order = False
        self.interval_start = None
        self.contact = torch.zeros(len(origins), 1, device=origins.device, dtype=torch.bool)
        self.contact_samples = 0

    def configure(self, sensor_origins, axis, modes, *, half_period_s=None, departure_delay_s=None, exact_clock=False):
        if self.enabled:
            raise RuntimeError('Existing moving-sphere block is active')
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
        travel_time, _ = self._travel_clock()
        return self.start + travel_time.minimum(self.end_travel_s)[:, None] * self.velocity

    def reset(self, indices):
        if self.enabled and len(indices):
            self.age[indices] = 0.
            if self.motion_ticks is not None:
                self.motion_ticks[indices] = 0
            self.servo_velocity[indices] = 0.
            self._place(self._pose()[indices], indices)

    def _place(self, position, indices=None):
        quaternion = position.new_zeros((len(position), 4))
        quaternion[:, 0] = 1.
        self.asset.write_root_pose_to_sim(torch.cat((position, quaternion), dim=1), env_ids=indices)
        self.asset.write_root_velocity_to_sim(position.new_zeros((len(position), 6)), env_ids=indices)

    def push(self, dt):
        if not self.enabled:
            return
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
        travel_time, direction = self._travel_clock()
        direction *= travel_time < self.end_travel_s
        self.latest = dict(tick=tick, applied_age_s=self.age.clone(), active=self.active.clone(),
                           sphere_position_w=center, expected_position_w=pose, sphere_velocity_w=velocity,
                           expected_velocity_w=direction[:, None] * self.velocity, servo_velocity_w=self.servo_velocity.clone(),
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
