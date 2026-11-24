"""Independent planar ray and swept-motion benchmark for local velocity control."""
from dataclasses import dataclass
import math

import torch


DT = .1
STEPS = 80
BEAMS = 90
HISTORY = 4
ANGLES = torch.arange(BEAMS) * (2 * math.pi / BEAMS)
DIRECTIONS = torch.stack((ANGLES.cos(), ANGLES.sin()), dim=-1)
RADIUS = .22
LOOK = 6.
SCENES = ('door', 'grove', 'street', 'pipe')
MOTIONS = ('clear', 'closing', 'opening', 'cross', 'hover', 'hover_clear')
# Fractions of the operator request plus lateral or rearward maneuver in m/s.
ACTIONS = torch.tensor([
    [1., 0., 0.], [.65, 0., 0.], [.33, 0., 0.], [0., 0., 0.],
    [.85, 0., .35], [.85, 0., -.35], [.60, 0., .52], [.60, 0., -.52],
    [0., 0., .55], [0., 0., -.55], [0., -.50, 0.],
    [.85, 0., .50], [.85, 0., -.50],
])


def rotate(vector, radians):
    c, s = radians.cos(), radians.sin()
    x, y = vector.unbind(-1)
    return torch.stack((c*x-s*y, s*x+c*y), -1)


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def action_vectors(command, previous, bins=ACTIONS):
    # Zero command retains a body-frame basis for evacuation.
    speed = command.norm(dim=-1, keepdim=True)
    basis = torch.where(speed > .01, command / speed.clamp_min(.01),
                        command.new_tensor([1., 0.]).expand_as(command))
    tangent = torch.stack((-basis[..., 1], basis[..., 0]), -1)
    return ((speed[:, None] * bins[:, 0, None] + bins[:, 1, None]) * basis[:, None]
            + bins[:, 2, None] * tangent[:, None])


@dataclass
class Observation:
    ranges: torch.Tensor
    valid: torch.Tensor
    acquisition: torch.Tensor
    clock: torch.Tensor
    gyro: torch.Tensor
    command: torch.Tensor
    previous: torch.Tensor
    command_history: torch.Tensor | None = None
    previous_history: torch.Tensor | None = None


class Arena:
    def __init__(self, *, seed, repeat, split, sensor):
        rng = torch.Generator().manual_seed(seed)
        self.codes = [(s, m, i) for s in range(4) for m in range(len(MOTIONS)) for i in range(repeat)]
        self.n = len(self.codes)
        scene, motion, replicate = torch.tensor(self.codes).T
        self.scene, self.motion = scene, motion
        self.sensor = sensor
        self.rng = rng
        sample = lambda *shape: torch.rand(*shape, generator=rng)
        noise = lambda *shape: sample(*shape) * 2 - 1
        self.position = torch.stack((.18 * noise(self.n), .18 * noise(self.n)), -1)
        self.start = self.position.clone()
        self.velocity = torch.zeros_like(self.position)
        self.previous = self.velocity.clone()
        self.yaw_start = sample(self.n) * 2 * math.pi
        self.yaw_rate = noise(self.n) * {'train': .24, 'development': .32, 'confirmation': .44}[split]
        held = float(split == 'confirmation')
        self.width = torch.tensor([2.1, 3.5, 2.95, 1.34])[scene] + .16 * noise(self.n) + .07*held
        self.speed = .58 + .11 * noise(self.n) + .035*held
        side = torch.where(replicate % 2 == 0, 1., -1.)
        v = .65 + .18 * sample(self.n)
        obj_x = torch.where(motion == 1, 3.55, torch.where(motion == 4, 2.45, 1.9))
        obj_y = torch.where(motion == 3, -side * v * (2.55 + .48 * noise(self.n)), .12 * noise(self.n))
        x_speed = torch.where(motion == 2, .72 + .22*sample(self.n),
                              torch.where((motion == 1) | (motion == 4), -.56 - .16*sample(self.n), 0.))
        y_speed = torch.where(motion == 3, side * v, 0.)
        mobile = torch.stack((obj_x + .26*noise(self.n), obj_y), -1)
        mobile[(motion == 0) | (motion == 5)] = 72.
        static = torch.tensor([[1.4, 1.12], [2.5, -1.22], [3.8, 1.07], [4.6, -.99]]).expand(self.n, -1, -1).clone()
        static += .26 * noise(self.n, 4, 2)
        static[scene == 0] += torch.tensor([0., .52])
        static[scene == 2] += torch.tensor([0., .62])
        static[scene == 3] = 70.
        self.centers = torch.cat((mobile[:, None], static), 1)
        self.motion_velocity = torch.zeros_like(self.centers)
        self.motion_velocity[:, 0] = torch.stack((x_speed, y_speed), -1)
        self.radii = (.17 + .14 * sample(self.n, 5))
        self.radii[:, 0] += .06 * held
        self.door_x = 2.9 + .35*noise(self.n) + .28*held
        self.slot = .77 + .13*noise(self.n)
        x, w, h = self.door_x, self.width, self.slot
        limit = torch.full_like(w, 90.)
        panel_x = torch.where(scene == 0, x, limit)
        self.segments = torch.stack((
            torch.stack((-limit, w, limit, w), 1),
            torch.stack((-limit, -w, limit, -w), 1),
            torch.stack((panel_x, -w, panel_x, -h), 1),
            torch.stack((panel_x, h, panel_x, w), 1)), 1).reshape(self.n, 4, 2, 2)
        self.live = torch.ones(self.n, dtype=torch.bool)
        self.raw = None
        self.delivered = None
        self.histories = None

    def command(self, t):
        active = (self.motion < 4) & (t < 5.5)
        return torch.stack((active * self.speed, torch.zeros_like(self.speed)), -1)

    def yaw(self, t):
        return self.yaw_start + self.yaw_rate*t

    def scan(self, t):
        directions = rotate(DIRECTIONS[None], self.yaw(t)[:, None])
        relative = (self.centers + self.motion_velocity*t) - self.position[:, None]
        b = torch.einsum('nrd,nod->nro', directions, relative)
        squared = relative.square().sum(-1) - self.radii.square()
        disc = b.square() - squared[:, None]
        near = b - disc.clamp_min(0.).sqrt()
        target = torch.where((near > 0) & (disc >= 0), near, torch.inf).amin(-1)
        segment = self.segments[:, :, 1] - self.segments[:, :, 0]
        origin = self.segments[:, :, 0] - self.position[:, None]
        denom = cross(directions[:, :, None], segment[:, None])
        denom = torch.where(denom.abs() < 1e-7, torch.nan, denom)
        scale = cross(origin, segment)[:, None] / denom
        along = cross(origin[:, None], directions[:, :, None]) / denom
        depth = torch.where((scale > 0) & (along >= 0) & (along <= 1), scale, torch.inf).amin(-1)
        hit = torch.minimum(target, depth)
        valid = hit <= LOOK
        return hit.clamp(.05, LOOK), valid

    def observe(self, step):
        t = step * DT
        depth, valid = self.scan(t)
        t_col = torch.full((self.n, 1), t)
        profile = torch.arange(self.n) % 3
        if self.sensor:
            depth = (depth + torch.randn(depth.shape, generator=self.rng)*.009*self.sensor).clamp(.05, LOOK)
            mask = torch.rand(depth.shape, generator=self.rng) >= .015*self.sensor
            valid = valid & (mask | (profile == 0)[:, None])
        fresh = (torch.where(valid, depth, LOOK), valid, t_col)
        delayed = (profile >= 1) & (self.sensor > 0)
        drop = (profile == 2) & (self.sensor > 0) & (step > 0) & (step % 17 == 0)
        val = fresh if self.raw is None else tuple(torch.where(delayed[:, None], a, b) for a,b in zip(self.raw, fresh))
        if self.delivered is not None:
            val = tuple(torch.where(drop[:, None], a, b) for a,b in zip(self.delivered, val))
        self.raw, self.delivered = fresh, val
        yaw = self.yaw(t)
        command, previous = rotate(self.command(t), -yaw), rotate(self.previous, -yaw)
        reading = *val, t_col, self.yaw_rate[:, None], command, previous
        self.histories = [reading]*HISTORY if self.histories is None else self.histories[1:] + [reading]
        history = [torch.stack(v, 1) for v in zip(*self.histories)]
        return Observation(*history[:5], command, previous, *history[5:])

    def clearance(self, positions, times):
        n = self.n
        p = positions.reshape(n, -1, 2)
        t = torch.broadcast_to(times, positions.shape[:-1]).reshape(n, -1, 1, 1)
        center = self.centers[:, None] + self.motion_velocity[:, None] * t
        disc = ((p[:, :, None] - center).norm(dim=-1) - self.radii[:, None]).amin(-1)
        point = self.segments[:, None, :, 0]
        segment = self.segments[:, None, :, 1] - point
        diff = p[:, :, None] - point
        portion = ((diff*segment).sum(-1) / segment.square().sum(-1)).clamp(0., 1.)
        walls = (diff - portion[..., None]*segment).norm(dim=-1).amin(-1)
        return (torch.minimum(disc, walls) - RADIUS).reshape(positions.shape[:-1])

    def teacher(self, observed, t, horizon=1.5):
        actions = action_vectors(observed.command, observed.previous)
        world_actions = rotate(actions, self.yaw(t)[:, None])
        until = torch.linspace(.05, horizon, 14)
        # First-order identified speed response, integrated from the current velocity.
        lag = .22 * (1 - torch.exp(-until/.22))
        predictions = (self.position[:, None, None] + world_actions[:, :, None]*(until-lag)[None,None,:,None]
                       + self.velocity[:, None, None]*lag[None,None,:,None])
        clearance = self.clearance(predictions, t+until).amin(-1)
        error = (world_actions - self.command(t)[:, None]).square().sum(-1)
        error += .045*(world_actions - self.previous[:, None]).square().sum(-1)
        risk = ((.20-clearance)/.20).clamp(0., 1.)
        # Preserve safe requests while choosing a feasible active maneuver early.
        best = (error + 3.*risk + 3.*(clearance < 0) - .025*clearance.clamp(max=.25)).argmin(-1)
        return actions[torch.arange(self.n), best], risk, best

    def advance(self, body, t):
        target = rotate(body, self.yaw(t))
        old = self.position.clone()
        v = self.velocity + (target-self.velocity) * (1-math.exp(-DT/.22))
        delta = v*DT
        frac = torch.linspace(0., 1., 7)
        clearance = self.clearance(old[:, None] + delta[:, None]*frac[None,:,None], t+DT*frac).amin(1)
        # Exact relative linear sweep for the moving and static circular objects.
        relative = old[:, None] - self.centers - t*self.motion_velocity
        change = delta[:, None] - DT*self.motion_velocity
        tc = (-(relative*change).sum(-1)/change.square().sum(-1).clamp_min(1e-8)).clamp(0., 1.)
        exact = ((relative + tc[...,None]*change).norm(dim=-1) - self.radii - RADIUS).amin(1)
        clearance = torch.minimum(clearance, exact)
        active = self.live.clone()
        self.position += delta * active[:, None]
        self.velocity = v * active[:, None]
        self.previous = target
        self.live &= clearance > 0.
        return torch.where(active, clearance, torch.inf), active
