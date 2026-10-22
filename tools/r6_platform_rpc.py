"""Bounded read-only observations from this task's existing local app-server.

Uses only initialize and three documented read methods. Never starts a thread,
changes a goal, installs dependencies, or controls the daemon.
"""
import base64
import hashlib
import json
import math
import os
import socket
import stat
import struct
import time
from pathlib import Path
from rlflow.jobs import identity,owns
from tools.r6_recovery import RecoveryError,checked,file_pin

SOCKET_ALIAS=Path('/home/mint/.codex/app-server-control/app-server-control.sock')
# The managed daemon publishes a symlink. Pin its observed canonical target;
# never follow an arbitrary replacement link when deciding whether to dispatch.
SOCKET=Path('/tmp/codex-daemon-1000/def9984b377d65c3648803af84f7ea666b746830c2b1a15fe192557140e51afb')
EXECUTABLE=dict(path='/home/mint/.codex/packages/app-server-daemon/releases/0.163.0-x86_64-unknown-linux-musl/bin/codex',
    sha256='399ff69c431c066e40de318fba45c2b07f45a76b19939d247ee4828fc70c4215')
DAEMON_PROCESS=dict(pid=1969832,pgid=1969832,starttime='204933637',boot_id='fa257cc3-1204-4e6f-88a5-a8a2ea7743aa')
OBJECTIVE='pasted text file: /home/mint/.codex/attachments/4daf07e8-ea6c-4823-9e41-4dfa9630515a/pasted-text-1.txt. Read this file before continuing.'
MAX_BYTES=256*1024
TIMEOUT=10
READ_METHODS={'initialize','initialized','thread/goal/get','thread/read','thread/turns/list'}


def require(ok,reason):
    if not ok:raise RecoveryError('PLATFORM_RPC_'+reason)


def enabled(state):return state.get('goal_id')!='R6-CPU-TEST-FIXTURE' or 'platform_rpc_fixture' in state


def settings(state):
    from tools.r6_platform_wait import binding
    source=binding(state)
    if state.get('goal_id')=='R6-CPU-TEST-FIXTURE':
        config=state['platform_rpc_fixture']
        return dict(socket=config['socket'],executable=config['executable'],process=config['process'],objective=config['objective'],source=source)
    require(SOCKET_ALIAS.lstat().st_uid==os.getuid() and SOCKET_ALIAS.resolve()==SOCKET,'SOCKET_ALIAS_DRIFT')
    return dict(socket=str(SOCKET),executable=EXECUTABLE,process=DAEMON_PROCESS,objective=OBJECTIVE,source=source)


class Reader:
    def __init__(self,config):
        self.config=config;self.sock=None;self.buffer=b'';self.total=0
        self.deadline=time.monotonic()+TIMEOUT

    def __enter__(self):
        try:
            path=Path(self.config['socket']);st=path.lstat()
            require(path.resolve()==path and not path.is_symlink() and stat.S_ISSOCK(st.st_mode)
                and st.st_uid==os.getuid(),'SOCKET_OWNER')
            self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.settimeout(TIMEOUT)
            self.sock.connect(str(path))
            pid,uid,gid=struct.unpack('3i',self.sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
            require(uid==os.getuid() and uid==st.st_uid,'PEER_OWNER')
            process=identity(pid);require(process is not None and process==self.config['process'] and owns(process),'PEER_PROCESS')
            exe=file_pin(Path('/proc')/str(pid)/'exe')
            require(exe==self.config['executable'],'PEER_EXECUTABLE');checked(exe)
            self.peer=dict(process=process,uid=uid,gid=gid,executable=exe,socket_inode=st.st_ino,socket_device=st.st_dev)
            key=base64.b64encode(os.urandom(16)).decode()
            self.write(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
                'Sec-WebSocket-Key: '+key+'\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
            while b'\r\n\r\n' not in self.buffer:
                self.read_more();require(len(self.buffer)<=65536,'HEADER_CAP')
            head,self.buffer=self.buffer.split(b'\r\n\r\n',1);rows=head.decode('ascii').split('\r\n')
            require(rows[0]=='HTTP/1.1 101 Switching Protocols','UPGRADE_STATUS')
            fields={k.lower():v.strip() for k,v in (r.split(':',1) for r in rows[1:])}
            accept=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
            require(fields.get('upgrade','').lower()=='websocket' and 'upgrade' in [v.strip() for v in fields.get('connection','').lower().split(',')]
                and fields.get('sec-websocket-accept')==accept,'UPGRADE_HEADERS')
            return self
        except BaseException:
            if self.sock:self.sock.close()
            raise

    def __exit__(self,*args):self.sock.close()

    def remaining(self):
        remaining=self.deadline-time.monotonic();require(remaining>0,'DEADLINE')
        self.sock.settimeout(remaining)

    def write(self,data):self.remaining();self.sock.sendall(data)

    def read_more(self):
        self.remaining();data=self.sock.recv(65536);require(bool(data),'EOF')
        self.total+=len(data);require(self.total<=MAX_BYTES,'RESPONSE_CAP');self.buffer+=data

    def take(self,n):
        while len(self.buffer)<n:self.read_more()
        result,self.buffer=self.buffer[:n],self.buffer[n:];return result

    def frame(self,data,opcode=1):
        require(len(data)<65536,'REQUEST_CAP');mask=os.urandom(4);n=len(data)
        self.write(bytes([128|opcode,128|(n if n<126 else 126)])
            +(b'' if n<126 else struct.pack('!H',n))+mask+bytes(b^mask[i%4] for i,b in enumerate(data)))

    def send(self,value):
        require(value['method'] in READ_METHODS,'READ_ONLY_METHOD')
        self.frame(json.dumps(value,separators=(',',':')).encode())

    def receive(self,request_id):
        for _ in range(32):
            a,b=self.take(2);require(a&128 and not a&112 and not b&128,'FRAME_FLAGS')
            opcode=a&15;n=b&127
            if n==126:n=struct.unpack('!H',self.take(2))[0]
            elif n==127:n=struct.unpack('!Q',self.take(8))[0]
            require(n<=MAX_BYTES,'FRAME_CAP');data=self.take(n)
            if opcode==9:
                require(n<=125,'PING_CAP');self.frame(data,10);continue
            require(opcode==1,'FRAME_OPCODE');value=json.loads(data)
            require(isinstance(value,dict),'MESSAGE')
            if value.get('id')==request_id:
                require('result' in value and 'error' not in value,'SERVER_ERROR')
                return value
            require('id' not in value and 'method' in value,'UNEXPECTED_RESPONSE')
        raise RecoveryError('PLATFORM_RPC_NOTIFICATION_CAP')


def observe(state):
    config=settings(state);started=time.time();mono=time.monotonic_ns();responses=[]
    try:
        with Reader(config) as client:
            client.send(dict(id=1,method='initialize',params=dict(clientInfo=dict(name='r6_platform_wait_reader',version='1'),capabilities=dict(experimentalApi=True))))
            responses.append(client.receive(1));client.send(dict(method='initialized',params={}))
            thread=config['source']['session_id']
            for n,method,params in [(2,'thread/goal/get',dict(threadId=thread)),
                (3,'thread/read',dict(threadId=thread,includeTurns=False)),
                (4,'thread/turns/list',dict(threadId=thread,limit=1,sortDirection='desc',itemsView='notLoaded')),
                (5,'thread/goal/get',dict(threadId=thread))]:
                client.send(dict(id=n,method=method,params=params));responses.append(client.receive(n))
            require(owns(client.peer['process']),'PEER_EXITED')
            result=dict(schema_version='r6-platform-rpc/1',started_at=started,finished_at=time.time(),
                elapsed_ns=time.monotonic_ns()-mono,peer=client.peer,source=config['source'],socket=config['socket'],
                responses=responses,implementation=file_pin(__file__))
        validate(state,result)
        return result
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise RecoveryError('PLATFORM_RPC_READ_FAILED:'+type(error).__name__) from error


def validate(state,record):
    config=settings(state);source=config['source'];peer=record['peer']
    require(record['schema_version']=='r6-platform-rpc/1' and record['socket']==config['socket']
        and all(record['source'][k]==source[k] for k in ('session_id','repository'))
        and record['source']['anchor']['path']==source['anchor']['path'],'SOURCE')
    require(peer['executable']==config['executable'] and peer['process']==config['process'] and peer['uid']==os.getuid(),'PEER_BINDING')
    proc=peer['process'];require(set(proc)=={'pid','pgid','boot_id','starttime'} and type(proc['pid']) is int
        and proc['pid']>0 and type(proc['pgid']) is int and proc['pgid']>0
        and isinstance(proc['boot_id'],str) and bool(proc['boot_id'])
        and isinstance(proc['starttime'],str) and proc['starttime'].isdigit(),'PROCESS_IDENTITY')
    checked(record['implementation']);checked(peer['executable'])
    start,end=record['started_at'],record['finished_at']
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (start,end)) and start<=end
        and type(record['elapsed_ns']) is int and 0<=record['elapsed_ns']<=TIMEOUT*10**9
        and 0<=end-start<=TIMEOUT+2 and end<=time.time()
        and abs((end-start)-record['elapsed_ns']/10**9)<=2,'TIME')
    rows=record['responses'];require([r.get('id') for r in rows]==[1,2,3,4,5]
        and all('result' in r and 'error' not in r for r in rows),'RESPONSES')
    goals=[rows[i]['result']['goal'] for i in (1,4)];thread=rows[2]['result']['thread'];turns=rows[3]['result']['data']
    require(all(g['threadId']==source['session_id'] and g['objective']==config['objective'] for g in goals),'GOAL_IDENTITY')
    require(thread['id']==source['session_id'] and thread['cwd']==source['repository']
        and thread['path']==source['anchor']['path'],'THREAD_IDENTITY')
    require(len(turns)==1,'LATEST_TURN');turn=turns[0]
    require(isinstance(turn['id'],str) and bool(turn['id']) and turn['rootTurnId']==turn['id']
        and turn['itemsView']=='notLoaded' and turn['items']==[],'TURN_IDENTITY')
    return dict(goals=goals,thread=thread,turn=turn)


def idle(state,record):
    value=validate(state,record);turn=value['turn']
    require(all(g['status']=='paused' for g in value['goals']),'GOAL_AUTO_CONTINUATION_NOT_STOPPED')
    require(all(type(g.get(k)) in (int,float) and math.isfinite(g[k]) and g[k]>=0
        for g in value['goals'] for k in ('tokensUsed','timeUsedSeconds','updatedAt')),'GOAL_USAGE_REQUIRED')
    require(value['thread']['status']['type']=='idle' and turn['status']=='completed'
        and turn['error'] is None,'ROOT_NOT_IDLE')
    require(type(turn['startedAt']) in (int,float) and type(turn['completedAt']) in (int,float)
        and math.isfinite(turn['startedAt']) and math.isfinite(turn['completedAt'])
        and turn['startedAt']<=turn['completedAt']<=record['started_at'],'TURN_TIME')
    return turn


def pair_proves_inactivity(state,start,end,coverage,observed):
    if not start or not end or not coverage:return False
    try:
        first=idle(state,start);last=idle(state,end)
        require(start['peer']==end['peer'],'PEER_RESTART')
        require(first==last,'TURN_CHANGED')
        goals=[r['responses'][i]['result']['goal'] for r in (start,end) for i in (1,4)]
        require(all(all(g[k]==goals[0][k] for k in ('tokensUsed','timeUsedSeconds','updatedAt'))
            for g in goals),'GOAL_USAGE_CHANGED')
        require(start['finished_at']<=coverage['started_at']<coverage['finished_at']<=end['started_at'],'WINDOW')
        rows=[t for t in observed['turns'] if t['turn_id']==first['id']]
        require(len(rows)==1 and rows[0]['finished_at'] is not None
            and abs(rows[0]['started_at']-first['startedAt'])<2
            and abs(rows[0]['finished_at']-first['completedAt'])<2,'NATIVE_TURN_BINDING')
        return True
    except (RecoveryError,KeyError,TypeError,ValueError):return False
