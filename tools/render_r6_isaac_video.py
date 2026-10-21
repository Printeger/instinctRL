"""R6 Isaac Sim replay implementation, awaiting registered render-job integration.

No CLI or dispatch authority. A controller must first admit and reserve a render
job and own its watchdog/cleanup. Only the renderer edits this new USD stage;
recorded poses are displayed without policy inference or physics integration.
"""
import json
import os
from pathlib import Path

from tools.r6_recovery import checked, file_pin
from tools import r6_video_presentation as presentation


def render(extraction_pin, output, *, source_snapshot, asset_root, check):
    """Render all selected full clips. `check` is the admitted worker's guard."""
    check()
    clips = presentation.load(extraction_pin)
    source = Path(source_snapshot).resolve()
    paths = ('isaaclab-training/instinctlab/task/scene_h1.py',
             'isaaclab-training/instinctlab/assets/taslab_uav/taslab_uav.usd')
    scene_sources = [file_pin(source / p) for p in paths]
    for clip in clips:
        original = Path(clip['decoded']['clip']['source_snapshot'])
        for relative, expected in zip(paths, scene_sources):
            presentation.require(file_pin(original / relative)['sha256'] == expected['sha256'], 'SCENE_SOURCE')
        checked(clip['decoded']['clip']['checkpoint'])
    out = Path(output)
    presentation.require(out.is_absolute() and out.resolve() == out and not out.exists(), 'RENDER_OUTPUT')
    # Read-only frozen imports are supplied by the worker PYTHONPATH, not main.
    from training_workloads.unified_h1.asset_closure import validate_asset_closure
    closure = validate_asset_closure(Path(asset_root), 'f64a0561477efcc612761f6b3034f9612f2d3cb42832138e86ff7a4c18e1ef52')
    os.environ['ISAACSIM_ASSET_ROOT'] = str(asset_root)
    out.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 960, 'height': 640,
                         'renderer': 'RayTracedLighting', 'anti_aliasing': 2,
                         'multi_gpu': False, 'active_gpu': 0, 'hide_ui': True})
    try:
        return _render(app, clips, extraction_pin, out, source, scene_sources, closure, check)
    finally:
        app.close()  # Outer registered worker must enforce cleanup deadline.


def _render(app, clips, extraction_pin, out, source, scene_sources, closure, check):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdLux, Gf
    from instinctlab.task import scene_h1
    presentation.require(file_pin(scene_h1.__file__) == scene_sources[0], 'IMPORTED_SCENE')
    combined_scene_triangles = scene_h1.combined_scene_triangles
    torch.set_num_threads(2)
    omni.timeline.get_timeline_interface().stop()
    rep.orchestrator.set_capture_on_play(False)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    light = UsdLux.DomeLight.Define(stage, '/World/ReplayLight')
    light.CreateIntensityAttr(300)
    light.CreateColorAttr((.85,.91,1.))
    sun = UsdLux.DistantLight.Define(stage, '/World/ReplaySun')
    sun.CreateIntensityAttr(800)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set((25,-25,-35))
    points, indices, identities = combined_scene_triangles(torch.zeros((1,3)))
    for i,name in enumerate(('Floor','XWall','YWall')):
        mesh = UsdGeom.Mesh.Define(stage, '/World/'+name)
        mesh.CreatePointsAttr(points[4*i:4*i+4])
        mesh.CreateFaceVertexCountsAttr([3,3])
        mesh.CreateFaceVertexIndicesAttr([v-4*i for v in indices[6*i:6*i+6]])
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([[(.22,.26,.30),(.39,.46,.54),(.44,.50,.57)][i]])
    drone = UsdGeom.Xform.Define(stage, '/World/ReplayDrone')
    translate = drone.AddTranslateOp()
    orient = drone.AddOrientOp()
    asset = stage.DefinePrim('/World/ReplayDrone/Asset')
    asset.GetReferences().AddReference(str(source/'isaaclab-training/instinctlab/assets/taslab_uav/taslab_uav.usd'))
    def line(name, pts, color, width=.012):
        c=UsdGeom.BasisCurves.Define(stage, '/World/'+name)
        c.CreateTypeAttr('linear'); c.CreateCurveVertexCountsAttr([len(pts)])
        c.CreatePointsAttr(pts);c.CreateWidthsAttr([width]);c.SetWidthsInterpolation('constant')
        c.CreateDisplayColorAttr([color]); return c
    for j in range(-6,7):
        line(f'GridX{j+6}', [(j,-6,.005),(j,6,.005)], (.35,.40,.45),.006)
        line(f'GridY{j+6}', [(-6,j,.005),(6,j,.005)], (.35,.40,.45),.006)
    line('SafeX',[(5.2,-6,2.05),(5.2,6,2.05)],(1.,.18,.06),.018)
    line('SafeY',[(-6,5.2,2.05),(6,5.2,2.05)],(1.,.18,.06),.018)
    trail = line('Trail',[(0,0,2),(0,0,2)],(.1,.8,1),.008)
    camera = UsdGeom.Camera.Define(stage, '/World/ReplayCamera')
    camera.CreateFocalLengthAttr(28)
    camera.CreateClippingRangeAttr((.05,200.))
    cam_transform = camera.AddTransformOp()
    product = rep.create.render_product(str(camera.GetPath()), (960, 640))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    videos = []
    try:
        for index, item in enumerate(clips):
            check()
            clip = item['decoded']['clip']
            path = out / f'clip-{index:02d}.mp4'
            writer = imageio.get_writer(str(path), fps=10, codec='libx264', quality=8,
                macro_block_size=1, ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
            try:
                for presentation_frame in item['frames']:
                    check()
                    row = presentation_frame['recorded']
                    pos = np.array(row['position_local_m'])
                    q = row['quaternion_wxyz']
                    translate.Set(Gf.Vec3d(*pos))
                    orient.Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
                    eye = pos + np.array([-2.6, -3.4, 1.9])
                    target = pos + np.array([.3, .25, 0.])
                    cam_transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse())
                    history = presentation_frame['trail']
                    trail.GetPointsAttr().Set(history if len(history)>1 else history*2)
                    trail.GetCurveVertexCountsAttr().Set([max(2,len(history))])
                    rep.orchestrator.step(rt_subframes=4 if row['frame']==0 else 1, delta_time=0., pause_timeline=True)
                    data = rgb.get_data()
                    presentation.require(data.size and data.shape[:2] == (640,960), 'RENDER_SHAPE')
                    frame = Image.fromarray(data[:,:,:3].astype(np.uint8))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,0,960,96), fill=(14,21,30))
                    draw.rectangle((0,490,960,640), fill=(14,21,30))
                    draw.text((16,8), f'InstinctRL R6 | {clip["block_id"]} | lane {clip["lane"]}', font=font, fill='white')
                    draw.text((16,36), 'ISAAC SIM 3D | Recorded evaluation replay | No new dynamics', font=small, fill='#90cdf4')
                    draw.text((16,61), f'Checkpoint {clip["checkpoint"]["sha256"][:24]} | seed {clip["assignment"]["evaluation_seed"]}', font=small, fill='white')
                    draw.text((16,500), f't(state)={row["state_time_s"]:.1f}s | t(control)={row["control_time_s"]:.1f}s', font=font, fill='white')
                    velocity = ', '.join(f'{v:+.3f}' for v in row['velocity_w_m_s'])
                    command = ', '.join(f'{v:+.3f}' for v in row['command_w_m_s'])
                    draw.text((16,528), f'Actual v_world: [{velocity}] m/s', font=font, fill='#7de3ff')
                    draw.text((16,555), f'Pre-step command_world: [{command}] m/s', font=font, fill='white')
                    draw.text((16,582), f'Sensor geometry distance: {row["geometry_distance_m"]:.3f} m | wall safety line: 0.80 m', font=small, fill='#ffbb78')
                    draw.text((16,611), f'Collision={row["collision"]} | height violation={row["height_violation"]} | reset after state={row["reset_after_state"]}', font=small, fill='#ffd86b')
                    writer.append_data(np.asarray(frame))
                    if row['frame'] in (0, len(item['frames'])-1):
                        frame.save(out / f'clip-{index:02d}-frame-{row["frame"]:03d}.png')
            finally:
                writer.close()
            telemetry = out / f'clip-{index:02d}.json'
            telemetry.write_text(json.dumps(dict(trajectory=item['trajectory'], frames=item['frames']), allow_nan=False)+'\n')
            videos.append(dict(index=index, trajectory=item['trajectory'], video=file_pin(path), telemetry=file_pin(telemetry)))
            check()
        # A changed raw or extracted source invalidates this render, not just validation.
        presentation.load(extraction_pin)
        for pin in scene_sources:
            checked(pin)
        manifest = dict(schema_version='r6-video-render/1', status='RENDERED_PENDING_CPU_VALIDATION',
            extraction=extraction_pin, renderer=file_pin(__file__), scene_sources=scene_sources,
            asset_closure=closure, videos=videos, physics_steps=0, policy_inference_steps=0,
            optimizer_updates=0, checkpoint_writes=0, scientific_acceptance=False,
            pose_semantics='Recorded pre-reset post-step pose; no interpolation. Trail breaks after reset.',
            presentation='Frozen scene and UAV asset; added camera, lights, grid and HUD. Rotor phase unavailable.')
        (out/'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        return manifest
    finally:
        rgb.detach([product.path])
        product.destroy()
