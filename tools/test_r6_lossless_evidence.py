import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zlib
import torch

from training_workloads.unified_h1.lossless_evidence import EvidencePool,ChunkReader,CHUNK
from training_workloads.unified_h1.robustness_training_storage import R6LearningLog,R6FirstUpdateDiagnostic
from training_workloads.unified_h1.logging import LearningLog


class LosslessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def pool(self,maximum=16*1024**2,raw_maximum=8*1024**2):
        return EvidencePool(self.root/'compressed',run_id='CPU',maximum_bytes=maximum,maximum_raw_file_bytes=raw_maximum)

    def reader(self,name,limit=8*1024**2):
        return ChunkReader(self.root/'compressed'/(name+'.r6z'),run_id='CPU',logical_name=name,maximum_raw_bytes=limit)

    def test_chunk_boundaries_seek_readinto_and_exact_bytes(self):
        pool=self.pool();stream=pool.open('blob');data=(bytes(range(256))*(CHUNK//256+1))*3
        stream.write(data[:3]);stream.flush();stream.write(data[3:]);stream.close()
        with self.reader('blob') as reader:
            self.assertEqual(reader.read(),data)
            for offset in (0,2,3,CHUNK-2,CHUNK+2,len(data)-4):
                reader.seek(offset);self.assertEqual(reader.read(11),data[offset:offset+11])
            reader.seek(-7,io.SEEK_END);buffer=bytearray(7);self.assertEqual(reader.readinto(buffer),7);self.assertEqual(bytes(buffer),data[-7:])
        self.assertLess(pool.used,len(data));self.assertEqual(pool.used,stream.path.stat().st_size)

    def test_torch_bytes_and_tensor_reload(self):
        value={'values':torch.arange(700000,dtype=torch.float32),'name':'CPU'}
        expected=io.BytesIO();torch.save(value,expected)
        pool=self.pool();writer=pool.open('input.pt');torch.save(value,writer);writer.close()
        with self.reader('input.pt') as reader:
            self.assertEqual(reader.read(),expected.getvalue())
            reader.seek(0);loaded=torch.load(reader,map_location='cpu',weights_only=True)
        self.assertTrue(torch.equal(loaded['values'],value['values']))

    def test_raw_and_compressed_bounds_poison_pool(self):
        pool=self.pool(raw_maximum=10);writer=pool.open('raw')
        with self.assertRaisesRegex(ValueError,'RAW_QUOTA'):writer.write(b'x'*11)
        writer.abort()
        with self.assertRaises(ValueError):pool.open('another')

    def test_output_quota_before_data_write(self):
        pool=self.pool(maximum=100);writer=pool.open('blob');before=pool.used
        writer.write(bytes(range(256)))
        with self.assertRaisesRegex(ValueError,'OUTPUT_QUOTA'):writer.flush()
        writer.abort();self.assertEqual(writer.path.stat().st_size,before)

    def test_failed_prefix_not_complete_and_no_retry(self):
        pool=self.pool();writer=pool.open('blob');writer.write(b'evidence');writer.flush();writer.abort()
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.reader('blob')
        with self.assertRaisesRegex(ValueError,'ALREADY_OPENED'):pool.open('blob')
        with self.assertRaises(FileExistsError):self.pool()

    def test_wrong_identity_and_read_bound(self):
        pool=self.pool();writer=pool.open('blob');writer.write(b'hello');writer.close()
        with self.assertRaisesRegex(ValueError,'IDENTITY'):
            ChunkReader(writer.path,run_id='OTHER',logical_name='blob',maximum_raw_bytes=100)
        with self.assertRaisesRegex(ValueError,'CHUNK_BOUND'):self.reader('blob',limit=4)

    def test_truncation_append_and_corruption(self):
        pool=self.pool();writer=pool.open('blob');writer.write(b'abc'*100);writer.close();original=writer.path.read_bytes()
        for modified in (original[:-1],original+b'X',original[:-41]+bytes([original[-41]^1])+original[-40:]):
            writer.path.write_bytes(modified)
            with self.assertRaises((ValueError,zlib.error)):self.reader('blob')

    def test_paths_and_symlink_rejected(self):
        pool=self.pool()
        for name in ('../escape','/absolute','a/../b','./alias'):
            with self.assertRaises(ValueError):pool.open(name)
        (pool.root/'alias').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):pool.open('alias/output')

    def test_learning_log_bytes_equal_legacy_jsonl(self):
        pool=self.pool();plain=LearningLog(self.root/'plain');compressed=R6LearningLog(self.root/'logical',pool=pool)
        for i in range(10):
            row={'sequence':i,'value':torch.tensor([.1,float('nan')]),'negative':True}
            plain.append('failures.jsonl',row);compressed.append('failures.jsonl',row)
        compressed.close()
        with self.reader('learning_log/failures.jsonl') as reader:
            self.assertEqual(reader.read(),(self.root/'plain/failures.jsonl').read_bytes())

    def test_diagnostic_saver_preserves_tensor_and_rng(self):
        pool=self.pool();output=self.root/'pilot';output.mkdir()
        diagnostic=R6FirstUpdateDiagnostic(output,pool=pool)
        before=torch.get_rng_state().clone();value={'tensor':torch.arange(12).reshape(3,4)}
        diagnostic._save_tensor(value,output/'input.pt')
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertFalse((output/'input.pt').exists())
        with self.reader('pilot_diagnostic/input.pt') as reader:
            self.assertTrue(torch.equal(torch.load(reader,weights_only=True)['tensor'],value['tensor']))

    def test_actual_gate_rejects_drift_before_optimizer_with_compressed_storage(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'tests/training_workloads/test_pilot_diagnostic_torch.py'
        spec=importlib.util.spec_from_file_location('r6_gate_cpu_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        trainer,batch=module.fixture();batch.old_alpha[17,2]+=.1
        pool=self.pool();diagnostic=R6FirstUpdateDiagnostic(self.root/'pilot',pool=pool)
        before=module._digest(module._state(trainer));rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'probability/distribution gate failed'):diagnostic.before_update(trainer,batch)
        self.assertEqual(before,module._digest(module._state(trainer)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(pool.writers,{})

    def test_actual_retained_rollout_can_be_read_without_plain_copy(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'tests/training_workloads/test_pilot_diagnostic_torch.py'
        spec=importlib.util.spec_from_file_location('r6_gate_cpu_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        trainer,batch=module.fixture();pool=self.pool();diagnostic=R6FirstUpdateDiagnostic(self.root/'pilot',pool=pool)
        diagnostic.retain_rollout(1,batch,trainer)
        with self.reader('pilot_diagnostic/update_1_full_input.pt') as reader:
            restored=torch.load(reader,map_location='cpu',weights_only=True)
        self.assertTrue(torch.equal(restored['old_alpha'],batch.old_alpha))
        self.assertTrue(torch.equal(restored['policy']['state_vec'],batch.policy['state_vec']))
        self.assertFalse((self.root/'pilot/update_1_full_input.pt').exists())


    def test_fsync_failure_poison_and_close_all_descriptors(self):
        pool=self.pool();a=pool.open('a');b=pool.open('b')
        a.write(b'first');b.write(b'second')
        with mock.patch('training_workloads.unified_h1.lossless_evidence.os.fsync',side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError,'disk failure'):a.flush()
        self.assertTrue(pool.failed)
        with self.assertRaisesRegex(ValueError,'FAILED'):pool.close()
        self.assertTrue(a.stream.closed and b.stream.closed)
        self.assertFalse(a.complete or b.complete)
        for name in ('a','b'):
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.reader(name)

    def test_pool_close_failure_closes_other_stream_without_footer(self):
        pool=self.pool();a=pool.open('a');b=pool.open('b')
        a.write(b'first');b.write(b'second')
        with mock.patch('training_workloads.unified_h1.lossless_evidence.os.fsync',side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError,'disk failure'):pool.close()
        self.assertTrue(pool.closed and pool.failed)
        self.assertTrue(a.stream.closed and b.stream.closed)
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.reader('b')
        with self.assertRaisesRegex(ValueError,'FAILED'):pool.close()

    def test_context_preserves_training_failure_and_rejects_partial_stream(self):
        pool=self.pool()
        with self.assertRaisesRegex(RuntimeError,'training failed'):
            with pool:
                a=pool.open('a');a.write(b'last evidence');a.flush()
                raise RuntimeError('training failed')
        self.assertTrue(pool.closed and a.stream.closed)
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.reader('a')

    def test_successful_pool_closure_prevents_new_files(self):
        pool=self.pool()
        with pool:
            writer=pool.open('a');writer.write(b'complete')
        pool.close()
        self.assertTrue(writer.complete)
        with self.reader('a') as reader:self.assertEqual(reader.read(),b'complete')
        with self.assertRaisesRegex(ValueError,'CLOSED'):pool.open('new')

    def test_installed_global_quota_rejects_before_chunk_write(self):
        from training_workloads.scale_003 import persistence
        quota=persistence.OutputWriteQuota(self.root,100)
        with mock.patch.object(persistence,'_output_quota',quota):
            pool=self.pool();writer=pool.open('blob');before=pool.used
            writer.write(bytes(range(256)))
            with self.assertRaises(persistence.PersistenceError):writer.flush()
            pool.abort()
        self.assertEqual(writer.path.stat().st_size,before)
        self.assertEqual(quota.used,before)
        self.assertTrue(pool.failed and pool.closed)

    def test_closed_learning_log_rejects_new_names(self):
        pool=self.pool();journal=R6LearningLog(self.root/'log',pool=pool)
        journal.append('updates.jsonl',{'update':1});journal.close()
        with self.assertRaisesRegex(ValueError,'LOG_CLOSED'):journal.append('new.jsonl',{})
        pool.close()


if __name__=='__main__':unittest.main()
