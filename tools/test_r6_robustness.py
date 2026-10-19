import copy
import json
import math
from pathlib import Path
import unittest

from training_workloads.unified_h1.robustness import command, score_lane, score_block, surface_distance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json').read_text())
BLOCKS = {b['id']: b for b in PROTOCOL['blocks']}


def fixture(name='S-clean-low', lane=0):
    block = BLOCKS[name]
    position = [0., 0., 2.]
    if block['task']=='wall_modes':
        position[0 if lane<16 else 1] = 6-block['sensor_to_wall_distance_m']
    def row(i, velocity):
        return dict(time_s=i*.1, sensor_position_w_m=list(position),
                    velocity_w_m_s=list(velocity), episode_id=1, collision=False,
                    height_violation=False, missing=False)
    initial = row(0, [0,0,0]); rows=[]
    for i in range(block['steps']):
        cmd = command(block,lane,i)
        velocity = cmd
        if block['task']=='wall_modes' and (lane%16)//4 == 0 and i>=30:
            velocity=(0,0,0)
        position = [p+v*.1 for p,v in zip(position,velocity)]
        rows.append(dict(row(i+1,velocity), policy_step=i,command_w_m_s=cmd))
    return dict(schema_version='r6-motion-trace/1',run_id='CPU-FIXTURE',block_id=name,
                lane=lane,env_origin_w_m=[0,0,0],initial=initial,rows=rows)


class RobustnessTests(unittest.TestCase):
    def score(self,t): return score_lane(BLOCKS[t['block_id']],t['lane'],t,run_id='CPU-FIXTURE')

    def test_nominal_stop_and_reverse(self):
        for name in ('S-clean-low','S-clean-high','X-clean'):
            for lane in range(32): self.assertTrue(self.score(fixture(name,lane))['motion_pass'])

    def test_static_low_speed_cannot_pass_with_small_rmse(self):
        t=fixture()
        for r in t['rows']:r['velocity_w_m_s']=[0,0,0]
        self.assertFalse(self.score(t)['motion_pass'])

    def test_missing_lane_keeps_denominator(self):
        t=[fixture(lane=i) for i in range(31)]
        result=score_block(BLOCKS['S-clean-low'],t,run_id='CPU-FIXTURE')
        self.assertEqual((result['successes'],result['total']),(31,32))
        self.assertFalse(result['motion_pass'])
        self.assertEqual(result['subgroups']['d7']['total'],4)

    def test_duplicate_lane_rejected(self):
        with self.assertRaises(ValueError):
            score_block(BLOCKS['S-clean-low'],[fixture(),fixture()],run_id='CPU-FIXTURE')

    def test_initial_collision_and_transient_geometry_are_failures(self):
        for index in (None,0,120,329):
            t=fixture(); r=t['initial'] if index is None else t['rows'][index]
            r['collision']=True
            self.assertFalse(self.score(t)['motion_pass'])
        t=fixture();t['initial']['sensor_position_w_m'][2]=.79
        self.assertFalse(self.score(t)['motion_pass'])

    def test_all_time_height_after_hold_counts(self):
        t=fixture();t['rows'][-1]['sensor_position_w_m'][2]=2.21
        self.assertFalse(self.score(t)['motion_pass'])

    def test_stop_hold_not_enough_or_drifts(self):
        t=fixture();t['rows'].pop()
        self.assertFalse(self.score(t)['motion_pass'])
        t=fixture();t['rows'][200]['sensor_position_w_m'][1]=.31
        self.assertFalse(self.score(t)['motion_pass'])

    def test_stop_confirmation_requires_full_second_and_three_second_deadline(self):
        t=fixture()
        for i in range(100,121): t['rows'][i]['velocity_w_m_s']=[.09,0,0]
        self.assertFalse(self.score(t)['motion_pass']) # confirmation t=13.2 >13
        t['rows'][120]['velocity_w_m_s']=[0,0,0]
        self.assertFalse(self.score(t)['motion_pass']) # t=13.1
        t['rows'][119]['velocity_w_m_s']=[0,0,0]
        self.assertTrue(self.score(t)['motion_pass']) # t=13.0

    def test_seed_run_and_command_mismatch_do_not_relabel(self):
        t=fixture();t['run_id']='OTHER'
        self.assertFalse(self.score(t)['motion_pass'])
        t=fixture();t['rows'][0]['command_w_m_s']=[0,0,0]
        self.assertFalse(self.score(t)['motion_pass'])

    def test_nan_and_reset_and_missing_flag_fail(self):
        for field,value in [('velocity_w_m_s',[float('nan'),0,0]),('episode_id',2),('missing',True),('time_s',float('nan')),('policy_step',True)]:
            t=fixture();t['rows'][0][field]=value
            self.assertFalse(self.score(t)['motion_pass'])

    def test_wall_modes_and_subgroups(self):
        block=BLOCKS['W-middle']
        result=score_block(block,[fixture('W-middle',lane) for lane in range(32)],run_id='CPU-FIXTURE')
        self.assertTrue(result['motion_pass'],result)
        self.assertEqual(len(result['subgroups']),8)
        self.assertTrue(all(s['total']==4 for s in result['subgroups'].values()))

    def test_wall_cannot_stop_first_then_move_late(self):
        t=fixture('W-middle')
        for i,r in enumerate(t['rows']):
            r['sensor_position_w_m']=[4.5,0,2]
            r['velocity_w_m_s']=[.1 if 200<=i<211 else 0,0,0]
        self.assertFalse(self.score(t)['motion_pass'])

    def test_wall_sensor_geometry_not_reported_clearance(self):
        t=fixture('W-middle'); t['rows'][0]['sensor_position_w_m']=[5.21,0,2]
        t['rows'][0]['claimed_clearance_m']=2
        self.assertFalse(self.score(t)['motion_pass'])

    def test_finite_rectangles_and_origin_translation(self):
        self.assertAlmostEqual(surface_distance([8,8,13],[0,0,0]),3.)
        self.assertAlmostEqual(surface_distance([108,208,13],[100,200,0]),3.)

    def test_pulse_motion_is_not_force_execution_acceptance(self):
        result=self.score(fixture('P-clean'))
        self.assertTrue(result['motion_pass'])
        self.assertEqual(result['scientific_acceptance'],'REQUIRES_INDEPENDENT_INPUT_CHAIN_AUDIT')

    def test_declared_final_capacity_matches_blocks(self):
        new=sum(b['steps']*b['lanes'] for b in PROTOCOL['blocks'])
        self.assertEqual(4*(new+32960),774400)
        self.assertEqual(PROTOCOL['capacity']['final_three_models_plus_baseline'],774400)


if __name__=='__main__': unittest.main()
