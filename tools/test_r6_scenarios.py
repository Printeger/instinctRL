import copy
import json
import math
from pathlib import Path
import unittest
from training_workloads.unified_h1.robustness_scenarios import reset_spec,verify_reset,pulse_force,rotate_yaw
from training_workloads.unified_h1.robustness import surface_distance

ROOT=Path(__file__).resolve().parents[1]
BLOCKS={b['id']:b for b in json.loads((ROOT/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json').read_text())['blocks']}


class ScenarioTests(unittest.TestCase):
    def spec(self,name='W-near',lane=0,seed=17,split='development'):
        return reset_spec(BLOCKS[name],seed=seed,split=split,lane=lane,mount_position_m=[.12,.07,.03])

    def test_mount_rotation_and_both_wall_distances(self):
        for name in ('W-near','W-middle','W-far'):
            for lane in range(32):
                s=self.spec(name,lane);offset=rotate_yaw(s['mount_position_m'],math.radians(s['yaw_degrees']))
                reconstructed=[p+d for p,d in zip(s['root_position_local_m'],offset)]
                for a,b in zip(reconstructed,s['sensor_position_local_m']):self.assertAlmostEqual(a,b)
                self.assertAlmostEqual(surface_distance(reconstructed,[0,0,0]),BLOCKS[name]['sensor_to_wall_distance_m'])
                self.assertTrue(verify_reset(s,s))

    def test_actual_root_position_cannot_substitute_sensor_distance(self):
        s=self.spec();bad=copy.deepcopy(s);bad['sensor_position_local_m']=bad['root_position_local_m']
        with self.assertRaises(ValueError):verify_reset(s,bad)

    def test_real_seed_and_split_changes_not_labels_only(self):
        a=self.spec('S-clean-low');b=self.spec('S-clean-low',seed=18);c=self.spec('S-clean-low',split='final')
        self.assertNotEqual(a['root_position_local_m'],b['root_position_local_m'])
        self.assertNotEqual(a['root_position_local_m'],c['root_position_local_m'])
        self.assertEqual(a,self.spec('S-clean-low'))

    def test_lane_reordering_does_not_change_draws(self):
        forward={i:self.spec(lane=i) for i in range(32)}
        reverse={i:self.spec(lane=i) for i in reversed(range(32))}
        self.assertEqual(forward,reverse)

    def test_reset_velocity_and_attitude_mismatch_fail(self):
        s=self.spec();bad=copy.deepcopy(s);bad['root_linear_velocity_w_m_s'][0]=.01
        with self.assertRaises(ValueError):verify_reset(s,bad)
        bad=copy.deepcopy(s);bad['root_quaternion_wxyz']=[1,0,0,.1]
        with self.assertRaises(ValueError):verify_reset(s,bad)
        bad=copy.deepcopy(s);bad['root_quaternion_wxyz']=[-x for x in s['root_quaternion_wxyz']]
        self.assertTrue(verify_reset(s,bad))

    def test_force_duration_and_impulse(self):
        b=BLOCKS['P-clean'];mass=1.1815179230179638
        forces=[pulse_force(b,lane=0,physics_tick=t,mass_kg=mass) for t in range(1100)]
        self.assertEqual(sum(any(v for v in f) for f in forces),40)
        self.assertAlmostEqual(sum(f[0]*.005 for f in forces),mass*.1)
        self.assertEqual(forces[999],(0.,0.,0.));self.assertEqual(forces[1040],(0.,0.,0.))
        self.assertEqual(pulse_force(BLOCKS['S-clean-low'],lane=0,physics_tick=1010,mass_kg=mass),(0.,0.,0.))


if __name__=='__main__':unittest.main()
