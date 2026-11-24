"""Opt-in PhysX kinematic sphere and its corresponding MID360 return."""

import torch


RADIUS_M = .34
DYNAMIC_MODES = ('approach', 'recede', 'cross_left', 'cross_right', 'hover',
                 'hover_wall', 'static_travel', 'static_hover')


def spawn(scene):
    from isaaclab import sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    cfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/R6MovingSphere',
        spawn=sim_utils.SphereCfg(
            radius=RADIUS_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., -20.)))
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
    start[:, 2] = torch.where(active, start[:, 2], -20.)
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
        self.origins = origins
        self.indices = torch.arange(len(origins), device=origins.device, dtype=torch.int32)
        self.age = torch.zeros(len(origins), device=origins.device, dtype=torch.float32)
        self.enabled = False
        self.half_period_s = None
        self.pre_reset = None
        self.latest = None
        self.verified_order = False
        self.interval_start = None
        self.contact = torch.zeros(len(origins), 1, device=origins.device, dtype=torch.bool)
        self.contact_samples = 0

    def configure(self, sensor_origins, axis, modes, *, half_period_s=None):
        if self.enabled:
            raise RuntimeError('Existing moving-sphere block is active')
        if not self.verified_order:
            parked = self.origins.clone()
            parked[:, 2] -= 20.
            observed = self.asset.root_physx_view.get_transforms()[:, :3]
            if observed.shape != parked.shape or not bool((observed-parked).abs().amax() <= .003):
                raise RuntimeError('Moving-sphere lanes differ from Isaac scene origins')
            self.verified_order = True
        self.start, self.velocity, self.active = trajectories(sensor_origins, axis, modes)
        self.half_period_s = half_period_s
        self.enabled = True
        self.reset(self.indices.to(torch.long))

    def _pose(self):
        travel_time, _ = travel_clock(self.age, self.half_period_s)
        return self.start + travel_time[:, None] * self.velocity

    def reset(self, indices):
        if self.enabled and len(indices):
            self.age[indices] = 0.
            position = self._pose()[indices]
            quaternion = torch.zeros(len(position), 4, device=position.device, dtype=position.dtype)
            quaternion[:, 0] = 1.
            self.asset.write_root_pose_to_sim(torch.cat((position, quaternion), dim=1), env_ids=indices)

    def push(self, dt):
        if not self.enabled:
            return
        self.age.add_(dt)
        position = self._pose()
        quaternion = torch.zeros(len(position), 4, device=position.device, dtype=position.dtype)
        quaternion[:, 3] = 1.  # Native transforms use XYZW.
        self.asset.root_physx_view.set_kinematic_targets(
            torch.cat((position, quaternion), dim=1).contiguous(), self.indices)

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
        lane = self.indices.long()
        _, direction = travel_clock(self.age, self.half_period_s)
        self.latest = dict(tick=tick, applied_age_s=self.age.clone(), active=self.active.clone(),
                           sphere_position_w=center, expected_position_w=pose, sphere_velocity_w=direction[:, None] * self.velocity,
                           sensor_position_w=origin.clone(), winning_beams=overwrite.sum(1),
                           sphere_sensor_clearance_m=(center-origin).norm(dim=1)-RADIUS_M,
                           witness_beam=witnesses, witness_direction_w=directions[lane, witnesses],
                           witness_range_m=finite_range[lane, witnesses])
        if capture:
            self.pre_reset = self.latest
        return result

    def finish(self):
        if self.enabled:
            position = self.origins.clone()
            position[:, 2] -= 20.
            quaternion = torch.zeros(len(position), 4, device=position.device, dtype=position.dtype)
            quaternion[:, 0] = 1.
            self.asset.write_root_pose_to_sim(torch.cat((position, quaternion), dim=1))
            self.enabled = False
