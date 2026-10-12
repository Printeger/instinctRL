"""CPU counterfactual wrench tests with preserved actual partial-live geometry.

No simulation, rotor mutation, optimizer call or new live acceptance is performed.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from instinctlab.assets.taslab_uav.config import TASLAB_UAV
from instinctlab.control.lee_velocity_controller import taslab_mixer,taslab_velocity_control
from instinctlab.control.runtime_adapter import EXPECTED_BODY_NAMES,measured_control_plant
from training_workloads.unified_h1.b0_enhanced import rotor_wrench,quaternion_rotate,reconstruct_com

def fixture():
    source=json.loads(Path('docs/diagnostics/r5-readiness/measurement-r1b-mechanics-samples.json').read_text())
    row=source['samples']['100']['row']
    assert row['experiment_id']=='UNIFIED-H1-R5-READINESS-B0-MEASUREMENT-R1-20261012'
    assert reconstruct_com(row['bodies'])['total_mass_kg']==row['system']['total_mass_kg']
    bodies=row['bodies']
    body=torch.tensor([[b['world_position_m']+b['world_quaternion_wxyz']+b['world_linear_velocity_m_s']+b['world_angular_velocity_rad_s'] for b in bodies]])
    local=torch.tensor([[b['local_com_m']+b['local_com_quaternion_wxyz'] for b in bodies]])
    mass=torch.tensor([[b['mass_kg'] for b in bodies]])
    articulation=NS(body_names=EXPECTED_BODY_NAMES,root_physx_view=NS(get_masses=lambda:mass),data=NS(body_link_state_w=body,body_com_pose_b=local))
    reference=body[:,0].clone();reference[:,7:]=0.
    return row,articulation,reference,mass

def actual_wrench(row,thrusts):
    rotors=copy.deepcopy(row['rotors'])
    for i,rotor in enumerate(rotors):
        rotor['thrust_n']=float(thrusts[i])
        rotor['reaction_moment_nm']=-TASLAB_UAV.rotors.directions[i]*TASLAB_UAV.rotors.moment_constants[i]/TASLAB_UAV.rotors.force_constants[i]*float(thrusts[i])
    return rotor_wrench(rotors,row['system']['combined_com_world_m'])

def test_actual_mass_authority_and_geometry_remove_nominal_allocation_bias():
    row,articulation,reference,_=fixture()
    mass,allocation=measured_control_plant(articulation,reference)
    assert float(mass)==pytest.approx(row['system']['total_mass_kg'],abs=1e-7)
    desired=torch.tensor([0.,0.,0.,float(mass)*9.81])
    thrusts=torch.linalg.solve(allocation[0],desired)
    assert bool((thrusts>0).all())
    actual=actual_wrench(row,thrusts)
    assert max(abs(x) for x in actual['torque_at_com_nm'])<2e-6
    assert actual['force_n'][2]==pytest.approx(float(mass)*9.81,abs=2e-5)
    old=taslab_mixer(reference)@desired
    biased=actual_wrench(row,old)
    assert abs(biased['torque_at_com_nm'][1])>.02
    assert abs(biased['torque_at_com_nm'][0])>.005

def test_controller_compensation_uses_measured_whole_mass_without_changing_plant():
    row,articulation,reference,raw_mass=fixture();before=raw_mass.clone()
    mass,allocation=measured_control_plant(articulation,reference)
    current=taslab_velocity_control(reference,torch.zeros(1,3),measured_mass_kg=mass,measured_allocation=allocation)
    old=taslab_velocity_control(reference,torch.zeros(1,3))
    assert float(current.collective_thrust_n/old.collective_thrust_n)==pytest.approx(float(mass)/TASLAB_UAV.mass,abs=2e-7)
    assert torch.equal(before,raw_mass)
    with pytest.raises(ValueError,match='supplied together'):taslab_velocity_control(reference,torch.zeros(1,3),measured_mass_kg=mass)

@pytest.mark.parametrize('kind',['missing','nonpositive','duplicate','shape'])
def test_missing_or_invalid_authority_fails_without_nominal_mass_fallback(kind):
    _,articulation,reference,mass=fixture()
    if kind=='missing':articulation.root_physx_view.get_masses=lambda:None
    if kind=='nonpositive':mass[0,4]=0.
    if kind=='duplicate':articulation.body_names=('base_link','rotor_0','rotor_1','rotor_2','rotor_2')
    if kind=='shape':articulation.root_physx_view.get_masses=lambda:mass[:,:4]
    with pytest.raises((ValueError,RuntimeError)):measured_control_plant(articulation,reference)
