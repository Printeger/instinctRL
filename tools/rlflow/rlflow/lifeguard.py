"""Independent, identity-bound cleanup after supervisor death or loss of control."""
import ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path
from .jobs import identity,owns
from .store import atomic,canonical,sha
from .policy import stopped


def group_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'pgid':int(fields[2]),'starttime':fields[19],
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError,ProcessLookupError):return None


def kill_group(record):
    if record and record['pid']==record['pgid'] and group_identity(record['pid'])==record:
        try:os.killpg(record['pgid'],signal.SIGKILL)
        except ProcessLookupError:pass


def enable_subreaper():
    """Keep orphaned double-fork descendants attached to the experiment owner."""
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(),"PR_SET_CHILD_SUBREAPER failed")


def _process_table():
    """Snapshot parentage and process identity from each single /proc stat read."""
    table={}
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            pid=int(entry.name)
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]!='Z':
                table[pid]={'ppid':int(fields[1]),'identity':{
                    'pid':pid,'pgid':int(fields[2]),'starttime':fields[19],
                    'boot_id':boot_id}}
        except (FileNotFoundError,ProcessLookupError,ValueError,IndexError):pass
    return table


def _ancestry_chain(pid,owner_pid,table):
    """Return pid..owner only when every observed parent is in this snapshot."""
    chain=[];seen=set()
    while pid not in seen:
        seen.add(pid);observation=table.get(pid)
        if observation is None:return None
        chain.append(pid)
        if pid==owner_pid:return chain
        pid=observation['ppid']
    return None


def record_descendants(directory,owner,known=None):
    """Persist only ancestry-derived identities that survive admission validation."""
    prior=sorted((known or []),key=lambda row:row['pid'])
    known={row['pid']:row for row in prior if owns(row)}
    table=_process_table();owner_observation=table.get(owner['pid'])
    descendants=set()
    if owner_observation and owner_observation['identity']==owner:
        descendants={owner['pid']};changed=True
        while changed:
            changed=False
            for pid,observation in table.items():
                if observation['ppid'] in descendants and pid not in descendants:
                    descendants.add(pid);changed=True
    # Admission uses a second coherent parentage/identity snapshot.  Comparing
    # the complete candidate-to-owner chain prevents an ancestor PID replacement
    # from lending its numeric PID to an otherwise unchanged child.
    admission=_process_table() if descendants else {}
    final=_process_table() if descendants else {}
    for pid in sorted(descendants-{owner['pid'],os.getpid()}):
        chain=_ancestry_chain(pid,owner['pid'],table)
        if not (chain and all(admission.get(member)==table[member] for member in chain)):
            continue
        # Re-read coherent parentage immediately before admission, then revalidate
        # pid/starttime/boot identity for every chain member.  A replacement of
        # any observed ancestor therefore cannot lend its numeric PID to a child.
        if (any(final.get(member)!=table[member] for member in chain)
                or any(identity(member)!=table[member]['identity'] for member in chain)):
            continue
        observed=table[pid]['identity']
        known[pid]=observed
    rows=sorted(known.values(),key=lambda row:row['pid'])
    ledger=Path(directory)/'descendant-ledger.json'
    if rows!=prior or not ledger.exists():
        atomic(ledger,canonical({'owner':owner,'descendants':rows,'updated_wall':time.time()}))
    return rows


def reclaim_descendants(directory,rows):
    actions=[]
    for sig,name in ((signal.SIGTERM,'SIGTERM'),(signal.SIGKILL,'SIGKILL')):
        for row in rows:
            matched=owns(row)
            if matched:
                try:os.kill(row['pid'],sig)
                except ProcessLookupError:matched=False
            actions.append({'identity':row,'signal':name,'identity_matched':matched})
        deadline=time.monotonic()+(.5 if sig==signal.SIGTERM else .2)
        while any(owns(row) for row in rows) and time.monotonic()<deadline:time.sleep(.01)
    evidence={'actions':actions,'survivors':[row for row in rows if owns(row)]}
    atomic(Path(directory)/'descendant-cleanup.json',canonical(evidence))
    return evidence


def start(directory,kind,owner,root,limit,object_id,spec=None):
    d=Path(directory);request={'directory':str(d),'kind':kind,'owner':owner,'root':str(root),
        'limit':limit,'id':object_id,'spec':spec,'started_wall':time.time(),'started_monotonic':time.monotonic()}
    path=d/'lifeguard-request.json';atomic(path,canonical(request))
    with (d/'lifeguard.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-m','rlflow.lifeguard',str(path)],
            cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    deadline=time.monotonic()+5
    while not (d/'lifeguard-ready.json').exists():
        if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('lifeguard failed before child launch')
        time.sleep(.01)
    ready=json.loads((d/'lifeguard-ready.json').read_text())
    if not owns(ready['identity']) or ready['request_sha256']!=sha(request):raise RuntimeError('lifeguard identity mismatch')
    return proc


def reclaim_descendants_until_quiescent(directory,owner,descendants,timeout=2,stable_scans=2):
    """Repeatedly discover and reclaim until ancestry is stably empty.

    A descendant can fork or detach after the completion snapshot.  Every scan
    remains rooted at the still-live owner, and every signal remains bound to a
    PID/starttime/boot identity.  Positive closure requires consecutive empty
    scans; exhausting the bound records failure evidence instead.
    """
    d=Path(directory);deadline=time.monotonic()+timeout;pending=list(descendants)
    actions=[];rounds=[];seen=[];quiet=0;timed_out=False
    while True:
        for row in pending:
            if row not in seen:seen.append(row)
        reclaimed=reclaim_descendants(d,pending)
        actions.extend(reclaimed['actions'])
        discovered=record_descendants(d,owner,pending)
        new=[row for row in discovered if row not in seen]
        for row in new:seen.append(row)
        survivors=[row for row in discovered if owns(row)]
        quiet=quiet+1 if not survivors and not new else 0
        rounds.append({'discovered':discovered,'new':new,'survivors':survivors,
                       'stable_scans':quiet})
        if quiet>=stable_scans:break
        if time.monotonic()>=deadline:
            timed_out=True;break
        pending=discovered
        time.sleep(.02)
    evidence={'actions':actions,'discovered':seen,'rounds':rounds,
              'survivors':rounds[-1]['survivors'],'stable_scans':quiet,
              'required_stable_scans':stable_scans,
              'quiescent':quiet>=stable_scans and not timed_out,'timed_out':timed_out}
    atomic(d/'descendant-cleanup.json',canonical(evidence))
    return evidence


def acknowledge_normal_completion(request,descendants):
    """Establish bounded ancestry quiescence before durable acknowledgement."""
    d=Path(request['directory'])
    cleanup=reclaim_descendants_until_quiescent(d,request['owner'],descendants)
    acknowledgement={'owner':request['owner'],'descendant_cleanup':cleanup,
                     'no_survivors':cleanup['quiescent'] and not cleanup['survivors']}
    atomic(d/'lifeguard-ack.json',canonical(acknowledgement))
    return acknowledgement


def await_normal_cleanup(directory,guard,owner,timeout=4):
    """Fail closed unless the guardian's durable cleanup evidence is complete."""
    d=Path(directory);guard.wait(timeout=timeout)
    acknowledgement=json.loads((d/'lifeguard-ack.json').read_text())
    cleanup=json.loads((d/'descendant-cleanup.json').read_text())
    if (acknowledgement.get('owner')!=owner or not acknowledgement.get('no_survivors')
            or acknowledgement.get('descendant_cleanup')!=cleanup or cleanup.get('survivors')
            or not cleanup.get('quiescent') or cleanup.get('timed_out')
            or cleanup.get('stable_scans',0)<cleanup.get('required_stable_scans',2)):
        raise RuntimeError('lifeguard normal-completion cleanup failed')
    return acknowledgement


def run(path):
    request=json.loads(Path(path).read_text());d=Path(request['directory']);owner=request['owner']
    atomic(d/'lifeguard-ready.json',canonical({'identity':identity(os.getpid()),'request_sha256':sha(request)}))
    cause='supervisor lost';descendants=[]
    while owns(owner):
        descendants=record_descendants(d,owner,descendants)
        if (d/'lifeguard-done').exists():
            descendants=record_descendants(d,owner,descendants)
            acknowledge_normal_completion(request,descendants)
            return
        if stopped(request['root']) or (d/'cancel').exists():cause='STOP/cancel';break
        if time.monotonic()-request['started_monotonic']>request['limit']:cause='supervisor deadline exceeded';break
        time.sleep(.02)
    reclaim(request,cause)

def reclaim(request,cause='supervisor lost during dispatcher recovery'):
    d=Path(request['directory']);owner=request['owner']
    ledger=json.loads((d/'descendant-ledger.json').read_text()).get('descendants',[]) if (d/'descendant-ledger.json').exists() else []
    if owns(owner):ledger=record_descendants(d,owner,ledger)
    child_file=d/({'role':'codex-identity.json','experiment':'process-identity.json','command':'command-child-identity.json'}[request['kind']])
    child=json.loads(child_file.read_text()) if child_file.exists() else None
    # Stop the owner before reading a possibly still-being-persisted identity.
    if owns(owner):
        os.kill(owner['pid'],signal.SIGTERM)
        deadline=time.monotonic()+.5
        while owns(owner) and time.monotonic()<deadline:
            # A cancellation/timeout can race the owner's completion handshake.
            # Finish identity-bound cleanup and acknowledge it instead of either
            # killing the owner mid-finally or treating cancellation as success.
            if (d/'lifeguard-done').exists():
                ledger=record_descendants(d,owner,ledger)
                acknowledge_normal_completion(request,ledger)
                return
            time.sleep(.01)
        if owns(owner):os.kill(owner['pid'],signal.SIGKILL)
    if child_file.exists():child=json.loads(child_file.read_text())
    kill_group(child)
    descendant_cleanup=reclaim_descendants(d,ledger)
    if (d/'receipt.json').exists():return
    deadline=time.monotonic()+2
    while (owns(owner) or owns(child)) and time.monotonic()<deadline:time.sleep(.01)
    elapsed=max(0,time.monotonic()-request['started_monotonic'])
    fact={'cause':cause,'owner':owner,'child':child,'owner_dead':not owns(owner),'child_alive':owns(child),'descendant_cleanup':descendant_cleanup,'elapsed_seconds':elapsed}
    atomic(d/('recovery-'+str(time.time_ns())+'.json'),canonical(fact))
    atomic(d/'recovery.json',canonical(fact))
    if owns(owner) or owns(child):return
    if request['kind']=='command':
        atomic(d/'receipt.json',canonical({'exit_code':137,'recovery':fact}))
        return
    if request['kind']=='role':
        from .accounting import observe
        released=(d/'model-release.json').exists()
        usage=observe(d,released)
        model_start=json.loads((d/'model-release.json').read_text()).get('started_monotonic',request['started_monotonic']) if released else None
        elapsed=max(0,time.monotonic()-request['started_monotonic'])
        active=max(0,elapsed-(model_start-request['started_monotonic'])) if released else 0
        receipt={'session_id':request['id'],'exit_code':137,'model_usage':usage,'model_active_seconds':active,
            'reason':'supervisor lost; owned child reclaimed','elapsed_seconds':elapsed,'finished':time.time(),'recovery':fact}
    else:
        release_path=d/'held-child.json';held=json.loads(release_path.read_text()) if release_path.exists() else {}
        released=bool(held.get('released_after_durable_record'))
        uncertain=(d/'release-intent.json').exists() and not released
        allocation=json.loads((d/'release-intent.json').read_text()).get('allocated_monotonic') if (d/'release-intent.json').exists() else None
        receipt={'job_id':request['id'],'spec_sha256':sha(request['spec']),'status':'STOPPED' if cause=='STOP/cancel' else 'FAILED',
            'exit_code':137,'error':cause,'elapsed_seconds':elapsed,'released':released,
            'definitely_not_released':not released and not uncertain,'recovery':fact,
            'allocated_seconds':max(0,time.monotonic()-allocation) if allocation is not None else None}
    atomic(d/'receipt.json',canonical(receipt))


def held_role(path):
    """No CLI executes until its own durable identity and release are bound."""
    r=json.loads(Path(path).read_text());d=Path(r['directory'])
    atomic(d/'codex-identity.json',canonical(group_identity(os.getpid())))
    deadline=time.monotonic()+10
    while not (d/'model-release.json').exists():
        if not owns(r['owner']) or time.monotonic()>deadline or stopped(r['root']):raise SystemExit(126)
        time.sleep(.01)
    release=json.loads((d/'model-release.json').read_text())
    if release['identity']!=group_identity(os.getpid()) or not owns(r['owner']) or stopped(r['root']) or (d/'cancel').exists():raise SystemExit(126)
    os.chdir(r['scratch'])
    os.execvpe(r['argv'][0],r['argv'],os.environ)


if __name__=='__main__':
    if sys.argv[1]=='held-role':held_role(sys.argv[2])
    else:run(sys.argv[1])
