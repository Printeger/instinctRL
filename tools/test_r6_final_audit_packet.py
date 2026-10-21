"""Real evidence reservation/artifacts; science and video provenance are explicit substitutes."""
from unittest.mock import Mock
import pytest
from tools import r6_final_audit_packet as m
from tools.test_r6_delivery_evidence import delivery, batch, run as deliver, FREE
from tools.r6_recovery import RecoveryError

@pytest.fixture
def ready(delivery,monkeypatch):
    b=delivery;b.store.replay=lambda:{};d=deliver(b)
    media=b.write('final-media.json',dict(visual_samples=[{'image':b.final['batch']}]))
    video=b.write('video-result.json',dict(manifest=media));receipt=b.write('video-receipt.json',dict(outputs={'result.json':video}))
    q={'fixture':'closed-video-inputs'}
    with b.j.writer():
        s=b.j.read();s['events']['video-validation']=dict(kind='cpu',identity={'role':'cpu-video-validation','event_id':'fixture-video'},request=dict(controller_key=b.controller,inputs=q),receipt=receipt)
        s['events'][b.controller]['cursor'].update(delivery_evidence=d,video_validation_key='video-validation',validated_videos=dict(status='VIDEOS_CPU_VALIDATED_REQUIRES_VISUAL_AUDIT',event_key='video-validation',receipt=receipt,result=video));b.j.save(s)
    monkeypatch.setattr(m.videos,'describe',Mock(return_value=q));monkeypatch.setattr(m.videos,'verified_receipt',Mock(return_value={'outputs':{'result.json':video}}))
    return b


def run(b):
    with b.j.writer():return m.prepare(b.j,b.controller,b.store,free_disk_bytes=FREE)


def test_post_video_packet_preserves_failures_unknown_waits_and_replays(ready):
    b=ready;out=run(b);budget=b.j.read()['budget'];assert run(b)==out and b.j.read()['budget']==budget
    assert out['status']=='FINAL_AUDIT_EVIDENCE_READY_NOT_ACCEPTED' and not out['goal_complete']
    pin=b.j.read()['events'][out['event_key']]['result']
    with b.j.writer():assert m.verify(b.j,pin,b.store)==out
    jobs=m.read(out['artifacts']['TRAINING_AND_EVALUATION_MANIFEST.json'])
    assert len(jobs['jobs'])==len(b.keys) and all(j['native_status']=='FAILED' for j in jobs['jobs'])
    models=m.read(out['artifacts']['MODEL_USAGE_AUDIT_PRE_REVIEW.json']);assert not models['platform_inactivity_proven']
    assert set(m.read(out['artifacts']['REQUIREMENTS_CHECKLIST.json'])['checks'])==set(m.CHECKS)


@pytest.mark.parametrize('fault',['stop','video_source','video_receipt','artifact','store'])
def test_packet_rejects_drift_and_stop(ready,fault):
    b=ready;out=run(b);pin=b.j.read()['events'][out['event_key']]['result']
    if fault=='stop':(b.j.directory/'STOP').touch()
    elif fault=='video_source':m.videos.describe.return_value={}
    elif fault=='video_receipt':m.videos.verified_receipt.return_value={'outputs':{'result.json':{}}}
    elif fault=='artifact':m.checked(out['artifacts']['RESOURCE_AUDIT_PRE_REVIEW.json']).write_text('{}')
    else:b.store.replay=lambda:{'changed':True}
    with b.j.writer(),pytest.raises(RecoveryError):m.verify(b.j,pin,b.store)
