import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.test_r6_robustness import BLOCKS, fixture
from training_workloads.unified_h1.robustness import score_block
from training_workloads.unified_h1.robustness_analysis import Inputs, replay_blocks, replay_original, summarize, encoded, analyze


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.block = copy.deepcopy(BLOCKS['S-clean-low'])
        self.folder = self.root/'r6'/self.block['id']
        self.folder.mkdir(parents=True)
        self.traces = [fixture(lane=i) for i in range(32)]
        self.write()

    def write(self):
        self.saved = dict(schema_version='r6-block-result/1', run_id='CPU-FIXTURE', seed=17,
            split='development', block_id=self.block['id'],
            scoring=score_block(self.block, self.traces, run_id='CPU-FIXTURE'))
        (self.folder/'motion_traces.json').write_bytes(encoded(self.traces))
        self.save_result()

    def save_result(self):
        (self.folder/'result.json').write_bytes(encoded(self.saved))

    def replay(self):
        return replay_blocks(self.root, [self.block], run_id='CPU-FIXTURE', seed=17, split='development')

    def test_complete_equal_is_not_scientific_acceptance(self):
        rows = self.replay()
        self.assertTrue(rows[0]['saved_result_equal'])
        summary = summarize(rows, {'replay_complete':False})
        self.assertFalse(summary['scientific_acceptance'])
        self.assertLess(len(encoded(summary)), 48*1024)
        self.assertEqual(summary['blocks'][0]['successes'], 32)

    def test_missing_lane_remains_failure(self):
        self.traces.pop(); self.write()
        row = self.replay()[0]
        self.assertTrue(row['saved_result_equal'])
        self.assertEqual((row['scoring']['successes'], row['total']), (31,32))
        self.assertEqual(summarize([row], {})['blocks'][0]['failed_lanes'], [31])

    def test_mismatched_saved_score(self):
        self.saved['scoring']['successes'] = 1; self.save_result()
        row = self.replay()[0]
        self.assertFalse(row['saved_result_equal'])
        self.assertEqual(row['error'], 'SAVED_SCORE_MISMATCH')

    def test_wrong_seed_is_evidence_error(self):
        self.saved['seed'] = 18; self.save_result()
        row = self.replay()[0]
        self.assertFalse(row['replay_complete'])
        self.assertIn('SAVED_RESULT_IDENTITY', row['error'])

    def test_missing_file_keeps_block_denominator(self):
        (self.folder/'motion_traces.json').unlink()
        row = self.replay()[0]
        self.assertEqual(row['total'],32)
        self.assertIsNone(row['scoring'])
        self.assertFalse(row['replay_complete'])

    def test_duplicate_lane_not_silently_dropped(self):
        self.traces.append(self.traces[0])
        (self.folder/'motion_traces.json').write_bytes(encoded(self.traces))
        self.assertIn('DUPLICATE_OR_UNKNOWN_LANE', self.replay()[0]['error'])

    def test_duplicate_block_rejected(self):
        with self.assertRaisesRegex(ValueError,'BLOCK_IDENTITIES'):
            replay_blocks(self.root,[self.block,self.block],run_id='CPU-FIXTURE',seed=17,split='development')

    def test_changed_input_rejected(self):
        reader=Inputs(); path=self.folder/'result.json'; reader.read(path)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'INPUT_CHANGED'):reader.verify()

    def test_symlink_rejected(self):
        path=self.folder/'alias.json';path.symlink_to(self.folder/'result.json')
        with self.assertRaisesRegex(ValueError,'ALIASED'):Inputs().read(path)

    def test_ambiguous_and_nonfinite_json_rejected(self):
        path=self.folder/'bad.json'
        for data in ('{"a":1,"a":2}', '{"a":NaN}'):
            path.write_text(data)
            with self.assertRaises(ValueError):Inputs().read(path)

    def test_input_limit(self):
        with self.assertRaisesRegex(ValueError,'INPUT_SIZE_BOUND'):
            Inputs().read(self.folder/'result.json', maximum=5)

    def test_summary_limit(self):
        with self.assertRaisesRegex(ValueError,'SUMMARY_SIZE_BOUND'):
            summarize(self.replay(), {'oversize':'a'*48*1024})

    def original_fixture(self):
        from training_workloads.unified_h1 import evaluation as ev
        raw=self.root/'evaluation_raw/r6_baseline_u640_original';raw.mkdir(parents=True)
        traces=[]
        for case in ev.CASE_ORDER:
            positions=ev.case_initial_positions(case).tolist()
            roots=[p+[1.,0.,0.,0.]+[0.]*6 for p in positions]
            rows=[dict(kind='T0',case=case,seed=20260909,env_origins_w_m=[[0.,0.,0.]]*32,
                initial_sensor_position_w_m=positions,initial_root_state_w=roots)]
            commands=ev.case_commands(case).tolist()
            for step in range(ev.CASES[case].steps):
                rows.append(dict(kind='STEP',case=case,step=step,command_b=commands[step],
                    actual_velocity_b=[[0.,0.,0.]]*32,sensor_positions_w_m=positions,episode_ids=[1]*32,
                    collision=[False]*32,below_height=[False]*32,above_height=[False]*32,
                    ray_clearance_m=[1.]*32,q=[False]*32,ics_beta=[1.]*32,root_state_w=roots))
            path=raw/(case+'.jsonl');path.write_bytes(b''.join(encoded(r) for r in rows))
            traces.extend(ev.replay_raw_case(str(path),case=case))
        (self.root/'r6_original_suite.json').write_bytes(encoded(ev.evaluate_checkpoint(traces,update=640)))
        (self.root/'r6_original_wall_motion.json').write_bytes(encoded(dict(per_lane=[False]*32,all32=False)))
        return raw

    def test_original_complete_replay_retains_performance_failure(self):
        self.original_fixture()
        result=replay_original(self.root)
        self.assertTrue(result['saved_result_equal'])
        self.assertTrue(result['saved_wall_equal'])
        self.assertFalse(result['scoring']['all_gates_pass'])
        self.assertEqual(len(result['scoring']['episodes']),160)

    def test_original_seed_and_extra_lane_rejected(self):
        raw=self.original_fixture();path=raw/'Z.jsonl'
        rows=[json.loads(r) for r in path.read_text().splitlines()]
        rows[0]['seed']=20261020
        path.write_bytes(b''.join(encoded(r) for r in rows))
        with self.assertRaisesRegex(ValueError,'ORIGINAL_SEED_MISMATCH'):replay_original(self.root)
        rows[0]['seed']=20260909;rows[1]['command_b'].append([0,0,0])
        path.write_bytes(b''.join(encoded(r) for r in rows))
        with self.assertRaisesRegex(ValueError,'ORIGINAL_LANE_DENOMINATOR'):replay_original(self.root)

    def test_combined_report_full_protocol_keeps_missing_blocks(self):
        self.original_fixture()
        protocol=Path(__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json'
        report, summary=analyze(self.root,protocol,run_id='CPU-FIXTURE',seed=17,split='development')
        self.assertEqual(len(report['blocks']),16)
        self.assertEqual(sum(b['replay_complete'] for b in report['blocks']),1)
        self.assertTrue(summary['original']['replay_complete'])
        self.assertFalse(report['protocol_frozen'])
        self.assertLess(len(encoded(summary)),48*1024)
        self.assertGreater(len(report['inputs']),8)

    def test_actual_workload_cpu_substitute_output_replays(self):
        from tools.test_r6_evaluation import EvaluationTests
        workload=EvaluationTests()
        backend=workload.setup_backend(self.root)
        workload.run_block(backend)
        rows=replay_blocks(self.root,[workload.block()],run_id='CPU-FIXTURE',seed=17,split='development')
        self.assertTrue(rows[0]['replay_complete'])
        self.assertTrue(rows[0]['saved_result_equal'])


if __name__ == '__main__': unittest.main()
