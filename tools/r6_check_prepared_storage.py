"""Read-only pre-spec validation of an already prepaid R6 production pair.

No provisioning, reservations, binding, Store permission or dispatch. Success
only supplies measured volume evidence for the later full capacity review.
"""
import argparse
import json
from pathlib import Path

from tools.r6_recovery import Journal,file_pin,checked
from tools.r6_storage_allocation import validate,require,native_tool,result_tool


def inspect(journal,key):
    entry,native,result,cap=validate(journal,key)
    pins={};missing=[]
    for label,plan,tool in [('native_volume',native,native_tool),('result_volume',result,result_tool)]:
        path=Path(plan['image']).parent/'volume.json'
        require(not path.is_symlink() and path.resolve()==path,'MANIFEST_PATH')
        if not path.exists():
            missing.append(str(path));continue
        pin=file_pin(path);record=json.loads(checked(pin).read_text())
        require(all(record.get(k)==v for k,v in plan.items())
            and record.get('implementation_sha256')==file_pin(tool.__file__)['sha256']
            and record.get('status')=='MOUNT_CREATED_NOT_ACCEPTED','MANIFEST_PLAN')
        pins[label]=pin
    output=dict(schema_version='r6-prepared-storage-check/1',event_key=key,
        experiment_id=entry['request']['experiment_id'],output_cap_bytes=cap,
        manifests=pins,missing=missing,capacity_pass=False,dispatch_authorized=False)
    if missing:return dict(output,status='WAITING_FOR_ADMIN_PROVISION')
    from training_workloads.unified_h1.native_volume import validate as native_validate
    from training_workloads.unified_h1.result_volume import validate as result_validate
    n=native_validate(pins['native_volume'])
    r=result_validate(pins['result_volume'],output_cap_bytes=cap,
        expected_target=result['target'],minimum_free_bytes=result['minimum_free_bytes'])
    return dict(output,status='VOLUMES_VERIFIED_NOT_ACCEPTED',native=n,result=r)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('event_key')
    args=parser.parse_args(argv)
    print(json.dumps(inspect(Journal(args.directory),args.event_key),sort_keys=True))


if __name__=='__main__':main()
