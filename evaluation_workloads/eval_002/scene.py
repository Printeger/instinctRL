"""Order-scoped geometry and observation-boundary command adapter.

Geometry copied from product e438c4c: taslab_direct_env._spawn_static_ray_plane
and scenario_corner.WALL_POINTS_M / WALL_FACE_VERTEX_INDICES. No product patch.
Runtime imports are deliberately deferred until AppLauncher has initialized.
"""
from collections import OrderedDict
import hashlib
import json

MESH_PATH = '/World/MID360StaticScene/Eval002CombinedMesh'
WALL_PATH = '/World/Eval002WallColliders/Walls'
WALL_POINTS = ((6.,-42.,0.),(6.,-42.,8.),(6.,6.,8.),(6.,6.,0.),
               (-42.,6.,0.),(6.,6.,0.),(6.,6.,8.),(-42.,6.,8.))
WALL_INDICES = (0,1,2,0,2,3,4,5,6,4,6,7)
COMMANDS = ((0.,0.,0.),(.2,0.,0.),(-.2,0.,0.),(0.,.2,0.),
            (0.,-.2,0.),(.15,.15,0.),(-.15,-.15,0.),(0.,0.,0.))


def command_at(case, index):
    if case not in ('station','tracking') or type(index) is not int or not 0 <= index <= 256:
        raise ValueError('case/global schedule index mismatch')
    return (0.,0.,0.) if case == 'station' else COMMANDS[min(index,255)//32]


def geometry():
    coordinates = range(-42,43,2)
    points = [(float(x),float(y),0.) for y in coordinates for x in coordinates]
    indices = []
    for row in range(42):
        for col in range(42):
            ll = row*43+col
            indices.extend((ll,ll+1,ll+44,ll,ll+44,ll+43))
    n = len(points)
    return points + list(WALL_POINTS), indices + [n+i for i in WALL_INDICES]


def validate_geometry(points, indices):
    import numpy as np
    expected_points, expected_indices = geometry()
    p, i = np.asarray(points), np.asarray(indices)
    if p.shape != (1857,3) or i.shape != (10596,) or not np.array_equal(p,expected_points) or not np.array_equal(i,expected_indices):
        raise ValueError('combined floor/two-wall topology mismatch')
    triangles = p[i.reshape(-1,3)]
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    if not (np.array_equal(normals[:3528],np.tile([0.,0.,1.],(3528,1)))
            and np.array_equal(normals[3528:3530],np.tile([-1.,0.,0.],(2,1)))
            and np.array_equal(normals[3530:],np.tile([0.,-1.,0.],(2,1)))):
        raise ValueError('surface normal mismatch')
    return dict(points=1857,triangles=3532,indices=10596,
                floor_normal=[0.,0.,1.],x_wall_normal=[-1.,0.,0.],y_wall_normal=[0.,-1.,0.])


class CommandBoundaryMixin:
    """Only observation construction selects the next global command.

    step() advances the *schedule index* before delegating. The old _command_b
    and _policy remain unchanged through physics/reward; inherited observation
    generation then builds the next command once, including after auto-reset.
    """
    def _get_observations(self):
        import torch
        command = torch.tensor(command_at(self.diagnostic_case,self.schedule_index),
                               device=self.device,dtype=self._command_b.dtype).repeat(self.num_envs,1)
        self._command_b.copy_(command)
        return super()._get_observations()

    def step(self, action):
        if self.schedule_index >= 256:
            raise RuntimeError('diagnostic policy budget exhausted')
        self.schedule_index += 1
        return super().step(action)


def make_environment(case):
    from instinctlab.envs import TaslabDirectEnv, TaslabDirectEnvCfg
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom, UsdPhysics

    class DiagnosticEnv(CommandBoundaryMixin,TaslabDirectEnv):
        def _setup_scene(self):
            # Inherited construction creates the original floor and all actors.
            # Sensor initialization occurs after _setup_scene returns.
            super()._setup_scene()
            stage = sim_utils.get_current_stage()
            points,indices = geometry()
            mesh = UsdGeom.Mesh.Define(stage,MESH_PATH)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([3]*(len(indices)//3))
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateSubdivisionSchemeAttr('none')
            UsdGeom.Xform.Define(stage,'/World/Eval002WallColliders')
            walls = UsdGeom.Mesh.Define(stage,WALL_PATH)
            walls.CreatePointsAttr(list(WALL_POINTS))
            walls.CreateFaceVertexCountsAttr([3]*4)
            walls.CreateFaceVertexIndicesAttr(list(WALL_INDICES))
            walls.CreateSubdivisionSchemeAttr('none')
            UsdPhysics.CollisionAPI.Apply(walls.GetPrim())

    cfg = TaslabDirectEnvCfg()
    cfg.seed = 20260905
    cfg.scene.num_envs = 2
    cfg.ray_caster.mesh_prim_paths = [MESH_PATH]
    if (cfg.scene.env_spacing,cfg.sim.dt,cfg.decimation) != (4.,.005,20):
        raise ValueError('frozen spacing/physics mismatch')
    DiagnosticEnv.diagnostic_case = case
    DiagnosticEnv.schedule_index = 0
    return DiagnosticEnv(cfg)


def live_geometry(env):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import RayCaster
    from pxr import UsdGeom, UsdPhysics
    if env.cfg.ray_caster.mesh_prim_paths != [MESH_PATH]:
        raise ValueError('selected mesh path mismatch')
    mesh = RayCaster.meshes[MESH_PATH]
    points = mesh.points.numpy().tolist()
    indices = mesh.indices.numpy().tolist()
    topology = validate_geometry(points,indices)
    stage = sim_utils.get_current_stage()
    authored = UsdGeom.Mesh(stage.GetPrimAtPath(MESH_PATH))
    validate_geometry([list(v) for v in authored.GetPointsAttr().Get()],list(authored.GetFaceVertexIndicesAttr().Get()))
    if authored.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('combined mesh must not add a duplicate floor collider')
    walls = UsdGeom.Mesh(stage.GetPrimAtPath(WALL_PATH))
    if (tuple(tuple(float(x) for x in p) for p in walls.GetPointsAttr().Get()) != WALL_POINTS
        or tuple(walls.GetFaceVertexIndicesAttr().Get()) != WALL_INDICES
        or not walls.GetPrim().HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError('sensed/collision wall mismatch')
    inventory = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(('/World/MID360StaticScene','/World/Eval002WallColliders','/World/ground')):
            attrs = [(str(a.GetName()),str(a.Get())) for a in prim.GetAttributes()]
            inventory.append(dict(path=path,type=str(prim.GetTypeName()),collision=bool(prim.HasAPI(UsdPhysics.CollisionAPI)),attributes=attrs))
    payload=dict(mesh_path=MESH_PATH,points=points,indices=indices,topology=topology,inventory=inventory,
                 env_origins=env.scene.env_origins.detach().cpu().tolist(),
                 initial_root_state_w=env._robot.data.root_state_w.detach().cpu().tolist())
    payload['sha256'] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload


def initial_source(env):
    """Read actual sensor without advancing task source or observation history."""
    import torch
    from isaaclab.utils.math import quat_apply
    _,validity,_,_ = env._read_sensor_producers()
    ray = env._ray_caster
    n = ray.ray_starts.shape[1]
    quat = ray.data.quat_w.repeat(1,n)
    return OrderedDict((
        ('ray_starts_w',(quat_apply(quat,ray.ray_starts)+ray.data.pos_w.unsqueeze(1)).detach().clone()),
        ('ray_directions_w',quat_apply(quat,ray.ray_directions).detach().clone()),
        ('accepted_ray_hits_w',ray.data.ray_hits_w.detach().clone()),
        ('accepted_beam_valid',validity.detach().clone()),
        ('mesh_path',MESH_PATH),('maximum_distance_m',float(env.cfg.ray_caster.max_distance)),
        ('source_step',torch.ones(2,dtype=torch.long,device=env.device)),
        ('reset_mask',torch.zeros(2,dtype=torch.bool,device=env.device))))


def visibility(source, acquisition):
    import torch
    hits = source['accepted_ray_hits_w']
    valid = source['accepted_beam_valid']
    normals = acquisition.raw_surface_normals_w
    masks = [valid & (hits[:,:,2].abs()<=1e-5) & (normals[:,:,2]>0.999),
             valid & ((hits[:,:,0]-6.).abs()<=1e-5) & (normals[:,:,0]<-0.999),
             valid & ((hits[:,:,1]-6.).abs()<=1e-5) & (normals[:,:,1]<-0.999)]
    return dict(zip(('floor','x_wall','y_wall'),[m.sum(-1).cpu().tolist() for m in masks]))


def initial_gate(source, acquisition):
    import torch
    counts=visibility(source,acquisition)
    result=dict(surface_hits=counts,numerical_rank=acquisition.observability.numerical_rank[0].cpu().tolist(),
                singular_values=acquisition.observability.singular_values[0].cpu().tolist(),
                sigma_min=acquisition.observability.sigma_min[0].cpu().tolist())
    if any(min(v)<=0 for v in counts.values()) or result['numerical_rank'] != [3,3] or not all(x>0 and torch.isfinite(torch.tensor(x)) for x in result['sigma_min']):
        raise ValueError('FIXTURE_NOT_OBSERVABLE: '+repr(result))
    return result
