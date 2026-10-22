"""Real Unix WebSocket CPU peer, synthetic root turns/goals and job lifecycles."""
import base64,copy,hashlib,json,os,socket,struct,sys,threading,time
from pathlib import Path
import pytest
from rlflow.store import sha
from rlflow.jobs import identity
from tools import r6_platform_rpc as rpc,r6_platform_wait as platform
from tools.r6_recovery import RecoveryError,file_pin
from tools.test_r6_platform_wait import traced,batch,lines,event
# pytest's transitive fixture dependency is named delivery.
from tools.test_r6_delivery_evidence import delivery
from tools import r6_delivery_evidence as audit_module


class Server:
    def __init__(self,path,source):
        self.path=path;self.source=source;self.settings={};self.methods=[];self.done=threading.Event();self.errors=[]
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.socket.bind(str(path));self.socket.listen();self.socket.settimeout(.1)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def close(self):self.done.set();self.thread.join(2);self.socket.close()

    def run(self):
        while not self.done.is_set():
            try:conn,_=self.socket.accept()
            except socket.timeout:continue
            with conn:
                conn.settimeout(1)
                try:self.handle(conn)
                except (BrokenPipeError,ConnectionResetError,TimeoutError,EOFError):pass
                except Exception as error:self.errors.append(repr(error))

    def handle(self,c):
        buf=b''
        while b'\r\n\r\n' not in buf:
            data=c.recv(4096)
            if not data:raise EOFError()
            buf+=data
        key=next(row.split(': ',1)[1] for row in buf.decode().split('\r\n') if row.startswith('Sec-WebSocket-Key: '))
        accept=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        if self.settings.get('bad_upgrade'):accept='wrong'
        c.sendall(('HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: '+accept+'\r\n\r\n').encode())
        def take(n):
            data=b''
            while len(data)<n:
                chunk=c.recv(n-len(data))
                if not chunk:raise EOFError()
                data+=chunk
            return data
        while True:
            a,b=take(2);assert b&128;n=b&127
            if n==126:n=struct.unpack('!H',take(2))[0]
            mask=take(4);raw=take(n);request=json.loads(bytes(v^mask[i%4] for i,v in enumerate(raw)))
            method=request['method'];self.methods.append(method)
            if method=='initialized':continue
            if self.settings.get('silence'):time.sleep(.2);return
            if self.settings.get('oversize'):c.sendall(bytes([129,127])+struct.pack('!Q',rpc.MAX_BYTES+1));return
            if method=='initialize':result=dict(userAgent='CPU-PEER')
            elif method=='thread/goal/get':result=dict(goal=dict(threadId=self.source['session_id'],objective='CPU goal',status=self.settings.get('goal','paused'),tokensUsed=100,timeUsedSeconds=20,updatedAt=20))
            elif method=='thread/read':result=dict(thread=dict(id=self.source['session_id'],cwd=self.settings.get('cwd',self.source['repository']),path=self.source['anchor']['path'],status=dict(type=self.settings.get('status','idle'))))
            elif method=='thread/turns/list':result=dict(data=[dict(id=self.settings.get('turn','turn-1'),rootTurnId=self.settings.get('turn','turn-1'),items=[],itemsView='notLoaded',status=self.settings.get('turn_status','completed'),startedAt=10,completedAt=20,error=None)],nextCursor='fixture')
            else:raise AssertionError(method)
            value=dict(id=request['id'],result=result)
            if self.settings.get('server_error'):value=dict(id=request['id'],error=dict(message='fixture error'))
            data=json.dumps(value).encode();c.sendall(bytes([129,len(data) if len(data)<126 else 126])+(b'' if len(data)<126 else struct.pack('!H',len(data)))+data)


@pytest.fixture
def api(traced,tmp_path):
    b=traced;source=platform.binding(b.j.read());server=Server(tmp_path/'app.sock',source)
    with b.j.writer():
        state=b.j.read();state['platform_rpc_fixture']=dict(socket=str(server.path),executable=file_pin(sys.executable),process=identity(os.getpid()),objective='CPU goal');b.j.save(state)
    b.server=server
    try:yield b
    finally:server.close()


def records(b):
    start=rpc.observe(b.j.read());end=rpc.observe(b.j.read())
    start.update(started_at=25,finished_at=26);end.update(started_at=45,finished_at=46)
    return start,end


def test_actual_socket_owner_process_and_only_read_methods(api):
    b=api;record=rpc.observe(b.j.read());turn=rpc.idle(b.j.read(),record)
    assert turn['id']=='turn-1' and record['peer']['process']['pid']==os.getpid()
    assert b.server.methods==['initialize','initialized','thread/goal/get','thread/read','thread/turns/list','thread/goal/get']
    assert not b.server.errors
    with pytest.raises(RecoveryError,match='READ_ONLY_METHOD'):rpc.Reader(rpc.settings(b.j.read())).send(dict(method='thread/goal/set'))


@pytest.mark.parametrize('settings,reason',[(dict(bad_upgrade=True),'UPGRADE_HEADERS'),(dict(oversize=True),'FRAME_CAP'),(dict(server_error=True),'SERVER_ERROR'),(dict(cwd='/wrong'),'THREAD_IDENTITY')])
def test_bad_transport_or_identity_never_becomes_idle_proof(api,settings,reason):
    api.server.settings.update(settings)
    with pytest.raises(RecoveryError,match=reason):rpc.observe(api.j.read())


def test_unresponsive_peer_has_total_read_deadline(api,monkeypatch):
    api.server.settings['silence']=True;monkeypatch.setattr(rpc,'TIMEOUT',.05)
    started=time.monotonic()
    with pytest.raises(RecoveryError,match='READ_FAILED'):rpc.observe(api.j.read())
    assert time.monotonic()-started<1


@pytest.mark.parametrize('settings,reason',[(dict(goal='active'),'AUTO_CONTINUATION'),(dict(status='active'),'ROOT_NOT_IDLE'),(dict(turn_status='inProgress'),'ROOT_NOT_IDLE')])
def test_active_goal_or_turn_cannot_dispatch(api,settings,reason):
    api.server.settings.update(settings);before=api.j.read()
    with api.j.writer(),pytest.raises(RecoveryError,match=reason):platform.preflight(api.j)
    assert api.j.read()==before


def test_exact_unchanged_live_turn_pair_brackets_wait(api):
    b=api;start,end=records(b);observed=platform.scan(lines(b.rows),platform.binding(b.j.read()))
    assert rpc.pair_proves_inactivity(b.j.read(),start,end,dict(started_at=30,finished_at=40),observed)


@pytest.mark.parametrize('damage',['restart','new_turn','wrong_goal','before_end','after_start','missing','stale_log','source','identity','usage','missing_usage'])
def test_incomplete_or_changed_pairs_remain_unknown(api,damage):
    b=api;start,end=records(b);observed=platform.scan(lines(b.rows),platform.binding(b.j.read()))
    if damage=='restart':end['peer']['process']['starttime']='999'
    if damage=='new_turn':end['responses'][3]['result']['data'][0].update(id='new',rootTurnId='new')
    if damage=='wrong_goal':end['responses'][4]['result']['goal']['status']='active'
    if damage=='before_end':end.update(started_at=35,finished_at=36)
    if damage=='after_start':start.update(started_at=35,finished_at=36)
    if damage=='missing':end=None
    if damage=='stale_log':observed['turns'][0]['turn_id']='older'
    if damage=='source':end['source']['session_id']='another'
    if damage=='identity':end['peer']['process']['pid']=True
    if damage=='usage':end['responses'][4]['result']['goal']['tokensUsed']+=1
    if damage=='missing_usage':end['responses'][4]['result']['goal'].pop('timeUsedSeconds')
    assert not rpc.pair_proves_inactivity(b.j.read(),start,end,dict(started_at=30,finished_at=40),observed)


def test_capture_and_delivery_accept_only_full_bracketed_coverage(api,monkeypatch):
    b=api;start,end=records(b);monkeypatch.setattr(rpc,'observe',lambda state:copy.deepcopy(end))
    with b.j.writer():
        state=b.j.read()
        for key in b.keys:
            job=state['events'][key];job['submission_claim']=dict(job_id=job['identity']['job_id'],wait_coverage_start=30,platform_wait_start=start)
            job['closure']=b.write('rpc-closure-'+key+'.json',dict(event_key=key,job_id=job['identity']['job_id'],request_sha256=sha(job['request'])))
        b.j.save(state)
        for key in b.keys:platform.capture(b.j,key)
    result=audit_module.model_audit(b.j.read())
    assert result['platform_inactivity_proven'] and result['pure_wait_model_calls']==0
    assert all(w['platform_evidence']['trace_completeness_reason']=='LIVE_APP_SERVER_UNCHANGED_COMPLETED_TURN' for w in result['waits'])
    b.rows.extend([event('task_started',35,'late'),event('task_complete',39,'late')]);b.trace.write_bytes(lines(b.rows))
    with pytest.raises(RecoveryError,match='LATE_ACTIVITY_DRIFT'):audit_module.model_audit(b.j.read())


def test_socket_symlink_and_wrong_executable_reject_before_rpc(api):
    b=api
    with b.j.writer():
        s=b.j.read();s['platform_rpc_fixture']['executable']['sha256']='0'*64;b.j.save(s)
    with pytest.raises(RecoveryError,match='PEER_EXECUTABLE'):rpc.observe(b.j.read())
    alias=b.server.path.parent/'alias.sock';alias.symlink_to(b.server.path)
    with b.j.writer():
        s=b.j.read();s['platform_rpc_fixture']['socket']=str(alias);b.j.save(s)
    with pytest.raises(RecoveryError,match='SOCKET_OWNER'):rpc.observe(b.j.read())
