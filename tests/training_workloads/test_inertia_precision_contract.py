"""CPU float32 counterexamples, not actual failed live body measurements."""
import copy
import pytest
import torch
from training_workloads.unified_h1.b0_enhanced import (
    EvidenceError, _body_inertia, inertia_symmetry_tolerance, quaternion_matrix)

def test_rotated_float32_positive_inertia_is_not_rejected_or_rewritten():
    q=torch.tensor([.71,.11,.53,.45]);q/=q.norm()
    rotation=torch.tensor(quaternion_matrix(q.tolist()))
    raw=rotation@torch.diag(torch.tensor([.07,.08,.11]))@rotation.T
    assert torch.linalg.eigvalsh(raw).min()>0
    matrix=raw.tolist();before=copy.deepcopy(matrix)
    skew=max(abs(matrix[i][j]-matrix[j][i]) for i in range(3) for j in range(3))
    assert skew>1e-10 and skew<inertia_symmetry_tolerance(matrix)
    result=_body_inertia(matrix,0,{'name':'CPU_counterexample'})
    assert [list(row) for row in result]==before==matrix

@pytest.mark.parametrize('kind,branch',[
    ('asymmetric','INERTIA_ASYMMETRIC'),('zero','INERTIA_NONPOSITIVE_DIAGONAL'),
    ('negative','INERTIA_NONPOSITIVE_DIAGONAL'),('nan','INERTIA_SHAPE_OR_NONFINITE')])
def test_invalid_inertia_retains_exact_failure_branch_and_raw_input(kind,branch):
    matrix=[[.07,0.,0.],[0.,.08,0.],[0.,0.,.11]]
    if kind=='asymmetric':matrix[0][1]=.001
    if kind=='zero':matrix[0][0]=0.
    if kind=='negative':matrix[0][0]=-.07
    if kind=='nan':matrix[0][0]=float('nan')
    with pytest.raises(EvidenceError) as caught:_body_inertia(matrix,3,{'name':'rotor_2'})
    context=caught.value.measurement_failure
    assert context['failure_branch']==branch and context['body_index']==3
    assert context['body_name']=='rotor_2' and context['raw_inertia'] is matrix

def test_precision_bound_scales_with_units_without_absolute_tolerance_floor():
    matrix=[[.07,0.,0.],[0.,.08,0.],[0.,0.,.11]]
    assert inertia_symmetry_tolerance([[x*1e-9 for x in row] for row in matrix])==pytest.approx(inertia_symmetry_tolerance(matrix)*1e-9,abs=0.)
