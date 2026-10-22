import base64,hashlib,json,os,socket,struct,time
from pathlib import Path
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(5)
sock.connect('/home/mint/.codex/app-server-control/app-server-control.sock')
peer=dict(zip(['pid','uid','gid'],struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))))
key=base64.b64encode(os.urandom(16)).decode();buf=b''
sock.sendall(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: '+key+'\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
while b'\r\n\r\n' not in buf:
 buf+=sock.recv(4096)
 if len(buf)>65536:raise RuntimeError('header cap')
headers,buf=buf.split(b'\r\n\r\n',1)
assert headers.startswith(b'HTTP/1.1 101'),headers[:100]
accept=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
assert accept in headers

def send(v,opcode=1):
 data=json.dumps(v).encode() if opcode==1 else v;mask=os.urandom(4);n=len(data)
 frame=bytes([128|opcode,128|(n if n<126 else 126)])+(b'' if n<126 else struct.pack('!H',n))+mask
 sock.sendall(frame+bytes(b^mask[i%4] for i,b in enumerate(data)))
def take(n):
 global buf
 while len(buf)<n:
  data=sock.recv(65536)
  if not data:raise RuntimeError('EOF')
  buf+=data
 value,buf=buf[:n],buf[n:];return value
def receive(expected):
 for _ in range(20):
  a,b=take(2);assert a&128,'fragmented response';n=b&127
  if n==126:n=struct.unpack('!H',take(2))[0]
  if n==127:n=struct.unpack('!Q',take(8))[0]
  assert n<=256*1024 and not b&128,'frame bound'
  data=take(n);opcode=a&15
  if opcode==9:send(data,10);continue
  assert opcode==1,(opcode,data[:100])
  r=json.loads(data)
  if r.get('id')==expected:return r
 raise RuntimeError('notification cap')
responses=[];thread='01a154cf-f2bb-79e3-8019-d6c78e8e0fc8';started=time.time()
try:
 send(dict(id=1,method='initialize',params=dict(clientInfo=dict(name='r6_readonly_handoff_audit',version='1.0'),capabilities=dict(experimentalApi=True))))
 responses.append(receive(1));print('initialize',responses[-1],flush=True)
 send(dict(method='initialized',params={}))
 for n,method,params in [(2,'thread/goal/get',dict(threadId=thread)),(3,'thread/read',dict(threadId=thread,includeTurns=False)),(4,'thread/turns/list',dict(threadId=thread,limit=1,sortDirection='desc',itemsView='notLoaded'))]:
  send(dict(id=n,method=method,params=params));responses.append(receive(n))
  r=responses[-1]
  if 'error' in r:print(r,flush=True)
  elif n==2:print('goal',r['result'],flush=True)
  elif n==4:print('latest_turn',r['result'],flush=True)
  else:
   t=r['result']['thread'];print('thread',{k:t.get(k) for k in ['id','status','path','cwd','updatedAt']},flush=True)
finally:
 sock.close();Path('/tmp/r6-platform-websocket-turns-probe.json').write_text(json.dumps(dict(started_at=started,finished_at=time.time(),thread_id=thread,methods=['initialize','initialized','thread/goal/get','thread/read','thread/turns/list'],peer_credentials=peer,responses=responses,model_call=False,goal_mutation=False),indent=2)+'\n')
