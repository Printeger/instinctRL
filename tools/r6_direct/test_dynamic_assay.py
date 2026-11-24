from types import SimpleNamespace

import torch

from instinctlab.observations.moving_sphere import MovingSphere, sphere_range, trajectories
from training_workloads.unified_h1.dynamic_assay import contract


def test_rays_choose_a_physical_near_surface_and_preserve_misses():
    origin = torch.tensor([[0., 0., 2.]])
    directions = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]]])
    center = torch.tensor([[1., 0., 2.]])
    depths = sphere_range(origin, directions, center)
    torch.testing.assert_close(depths, torch.tensor([[.66, torch.inf, torch.inf]]))
    hits = torch.tensor([[[2.,0.,2.], [0.,2.,2.], [-2.,0.,2.]]])
    transforms = torch.cat((center, torch.zeros(1,4)), dim=1)
    asset = SimpleNamespace(root_physx_view=SimpleNamespace(get_transforms=lambda: transforms,
                                                             get_velocities=lambda: torch.zeros(1, 6)))
    scene = MovingSphere(asset, origin)
    scene.start, scene.velocity, scene.active = center, torch.zeros_like(center), torch.tensor([True])
    out = scene.merge(hits, origin, directions, capture=True, tick=20)
    torch.testing.assert_close(out[0,0],torch.tensor([.66,0.,2.]))
    torch.testing.assert_close(out[0,1:],hits[0,1:])
    assert scene.pre_reset['winning_beams'].tolist() == [1]


def test_approach_and_crossing_paths_are_directional():
    expected = torch.tensor([[-.42,0.,0.],[.36,0.,0.],[0.,.58,0.],[0.,-.58,0.]])
    origin = torch.zeros(4,3)
    axis = torch.tensor([[1.,0.,0.]]).expand(4,-1)
    _, velocity, _ = trajectories(origin,axis,torch.arange(4))
    torch.testing.assert_close(velocity,expected)


def test_dynamic_result_budget_is_separate_and_exact(tmp_path):
    from tools.r6_evidence import file_pin

    first = tmp_path / 'candidate.pt'
    second = tmp_path / 'reference.pt'
    first.write_bytes(b'candidate')
    second.write_bytes(b'reference')
    params = dict(experiment_id='R6-DIRECT-INSTINCT-GPU-R2-DYNAMIC',training_seed=20260908,seed=1,
                  models=dict(candidate=file_pin(first),reference=file_pin(second)),
                  wall_seconds=720,output_cap_bytes=2**30)
    budgets = contract(params)['budgets']
    assert budgets['nontraining_transitions'] == 15360
    assert budgets['explicit_resets'] == 4
    assert budgets['optimizer_updates'] == 0
