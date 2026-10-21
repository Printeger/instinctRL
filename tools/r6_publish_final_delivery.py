"""Exact additive delivery commit/tag/push, retaining prior reports and evidence."""
from pathlib import Path
from rlflow.store import canonical
from tools.r6_recovery import checked,file_pin
from tools.r6_decision_packet import persist_exact
from tools import r6_final_delivery as delivery

read=delivery.read
require=delivery.require


def publish(journal,delivery_pin,store):
    result=delivery.verify(journal,delivery_pin,store);key=result['event_key'];entry=journal.read()['events'][key]
    request=entry['request'];root=delivery.ROOT
    require(delivery.git('symbolic-ref','--short','HEAD')=='main','PUBLICATION_MAIN')
    files=dict(result['files'],**{'DELIVERY_MANIFEST.json':delivery_pin})
    paths=sorted(str(checked(p).relative_to(root)) for p in files.values())
    directory=str(Path(result['folder']).relative_to(root))
    commits=delivery.git('log','-2','--format=%H','HEAD','--',directory).splitlines()
    require(len(commits)<=1,'PUBLICATION_HISTORY')
    if commits:commit=commits[0]
    else:
        require(delivery.git('rev-parse','HEAD')==request['base_commit'],'PUBLICATION_HEAD_DRIFT')
        delivery.profile.guard(journal);delivery.git('add','-f','--',*paths)
        delivery.profile.guard(journal);delivery.git('commit','--only','-m','R6 final delivery '+key,'--',*paths)
        commit=delivery.git('rev-parse','HEAD')
    require(delivery.git('show','-s','--format=%P',commit)==request['base_commit']
        and delivery.git('show','-s','--format=%s',commit)=='R6 final delivery '+key
        and delivery.git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()==paths,'PUBLICATION_COMMIT')
    for pin in files.values():
        path=str(Path(pin['path']).relative_to(root))
        require(delivery.snapshot.git(root,'show',commit+':'+path)==checked(pin).read_bytes(),'PUBLICATION_BYTES')
    tag='r6-final-delivery-'+key[:16]
    if delivery.git('tag','--list',tag):require(delivery.git('rev-parse',tag)==commit,'PUBLICATION_TAG')
    else:delivery.profile.guard(journal);delivery.git('tag',tag,commit)
    # Require the exact main head; do not push unrelated later local commits.
    require(delivery.git('rev-parse','HEAD')==commit,'PUBLICATION_HEAD_CHANGED')
    delivery.profile.guard(journal);delivery.git('push','origin','main','refs/tags/'+tag)
    refs=delivery.git('ls-remote','origin','refs/heads/main','refs/tags/'+tag)
    actual={row.split()[1]:row.split()[0] for row in refs.splitlines()}
    require(actual=={'refs/heads/main':commit,'refs/tags/'+tag:commit},'PUBLICATION_REMOTE')
    output=dict(status='FINAL_DELIVERY_PUBLISHED_PENDING_TERMINAL_AUDIT',event_key=key,delivery=delivery_pin,
        commit=commit,tag=tag,scientific_status=result['scientific_status'],goal_complete=False)
    data=canonical(output)+b'\n';require(len(data)<=65536,'PUBLICATION_RECEIPT_CAP')
    path=journal.directory/'final-publications'/key/'receipt.json';persist_exact(path,data)
    state=journal.read();state['events'][key]['publication']=file_pin(path);journal.save(state)
    return output
