"""Use a consistently observed surface to refine the direction of a crossing."""

import torch
from torch.nn import functional as F

from .escape_passage import rotate_scan_rays, track_nearby_return


def crossing_direction_from_planes(lidar, state, rays, motion):
    """Keep the original fit unless four scans place a sphere alongside a plane."""
    supported, _, times, shapes = track_nearby_return(
        lidar, state, rays, curved=True, maximum_previous_command=.46, fit_full_center=True)
    centers, radii, valid = (torch.stack(parts, dim=1) for parts in zip(*shapes, strict=True))
    history = state.reshape(-1, 4, 13)
    supported &= valid.all(dim=1) & (radii.amax(dim=1)-radii.amin(dim=1) <= .04)
    supported &= (history[:, :, 12] <= .04).all(dim=1)
    result = motion.clone()
    for lane in supported.nonzero().flatten():
        up = F.normalize(-history[lane, -1, 3:6], dim=0)
        direction = F.pad(motion[lane], (0, 1))
        direction = F.normalize(direction-(direction*up).sum()*up, dim=0)
        points = crossing_reference_points(lidar[lane], history[lane], rays, times[lane],
                                           centers[lane], radii[lane], up)
        aligned = supported_plane_direction(points, centers[lane], direction, up)
        if aligned is not None:
            result[lane] = aligned[:2]*(F.pad(motion[lane], (0, 1))*aligned).sum()
    return result


def crossing_reference_points(scan, history, rays, times, centers, radii, up):
    points = []
    grid = scan.flatten(1)
    aligned = rays
    for frame in (3, 2, 1, 0):
        if frame < 3:
            gyro = (history[frame, :3]+history[frame+1, :3])*.5
            aligned = rotate_scan_rays(aligned, -gyro*(times[frame+1]-times[frame]))
        ranges, valid, reliability = grid[3*frame:3*frame+3]
        hits = aligned*ranges[:, None]
        keep = (valid > .5) & (reliability > .55) & (ranges < 7.)
        keep &= (hits @ up).abs() <= .70
        keep &= (hits-centers[frame]).norm(dim=1) >= radii[frame]+.20
        points.append(hits[keep])
    return points


def supported_plane_direction(points, centers, heading, up):
    if any(len(frame) < 60 for frame in points):
        return None
    left = torch.cross(up, heading, dim=0)
    angle = heading.new_tensor(.005)*torch.arange(-44, 45, device=heading.device)
    normals = angle.cos()[:, None]*left+angle.sin()[:, None]*heading
    dots = normals @ points[0].T
    bins = ((dots+7.)/.025).floor().long().clamp(0, 559)
    counts = torch.zeros((len(normals), 560), device=heading.device, dtype=torch.long)
    counts.scatter_add_(1, bins, torch.ones_like(bins))
    offsets = (torch.arange(560, device=heading.device, dtype=heading.dtype)+.5)*.025-7.
    counts[:, offsets.abs() < 1.] = 0
    support, choices = counts.flatten().topk(24)
    candidates = support >= 60
    if not bool(candidates.any()):
        return None
    ids = choices[candidates]
    normals = normals[ids//560]
    anchor = offsets[ids%560]
    tangents = torch.cross(normals, up.expand_as(normals), dim=1)
    support = support[candidates]
    good = torch.ones_like(support, dtype=torch.bool)
    relative = []
    for frame, center in zip(points, centers.flip(0), strict=True):
        depth = normals @ frame.T
        delta = depth-anchor[:, None]
        nearby = delta.abs() < .15
        local = ((delta+.16)/.01).floor().long().clamp(0, 31)
        histogram = torch.zeros((len(normals), 32), device=heading.device, dtype=torch.long)
        histogram.scatter_add_(1, local, nearby.long())
        modes = anchor+histogram.argmax(dim=1)*.01-.155
        inside = (depth-modes[:, None]).abs() < .015
        count = inside.sum(dim=1)
        along = tangents @ frame.T
        height = (frame @ up).expand_as(depth)
        along_span = along.masked_fill(~inside, -torch.inf).amax(1)-along.masked_fill(~inside, torch.inf).amin(1)
        height_span = height.masked_fill(~inside, -torch.inf).amax(1)-height.masked_fill(~inside, torch.inf).amin(1)
        good &= (count >= 60) & (along_span >= .35) & (height_span >= .35)
        support = torch.minimum(support, count)
        mean = depth.masked_fill(~inside, 0.).sum(dim=1)/count.clamp_min(1)
        relative.append(normals @ center-mean)
    distances = torch.stack(relative, dim=1)
    good &= distances.amax(dim=1)-distances.amin(dim=1) <= .018
    if not bool(good.any()):
        return None
    index = support.masked_fill(~good, 0).argmax()
    tangent = tangents[index]
    return tangent*torch.sign(tangent @ heading)
