"""Inspect composed USD schema using existing libraries; no simulation or PhysX."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--usd-module-root',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing outputs')
    sys.path.insert(0,str(a.usd_module_root))
    from pxr import Usd,UsdPhysics,UsdGeom
    stage=Usd.Stage.Open(str(a.asset));bodies=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass=UsdPhysics.MassAPI(prim)
            bodies.append({'path':str(prim.GetPath()),'mass_kg':mass.GetMassAttr().Get(),
                           'mass_authored':mass.GetMassAttr().HasAuthoredValueOpinion(),
                           'com':str(mass.GetCenterOfMassAttr().Get()),'inertia':str(mass.GetDiagonalInertiaAttr().Get()),
                           'principal_axes':str(mass.GetPrincipalAxesAttr().Get())})
    layers=[]
    for layer in stage.GetUsedLayers():
        path=Path(layer.realPath)
        if layer.realPath and path.is_file():layers.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    unresolved=[b['path'] for b in bodies if not isinstance(b['mass_kg'],(int,float)) or b['mass_kg']<=0]
    result={'schema_version':'instinctrl-composed-usd-cpu/1','asset':str(a.asset.resolve()),
            'sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),'layers':layers,
            'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'kilograms_per_unit':UsdPhysics.GetStageKilogramsPerUnit(stage),
            'bodies':bodies,'total_runtime_mass_kg':None,'requires_runtime_mass_resolution':unresolved,
            'source':'composed USD schema CPU-only; zero/unset mass needs PhysX collider-derived mass, not zero summation',
            'simulation_initializations':0}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
