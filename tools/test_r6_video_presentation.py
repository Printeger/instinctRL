"""CPU presentation checks and actual MP4 decoding of synthetic images."""
import copy
import json
from pathlib import Path

import pytest

from tools.test_r6_video_trajectory import trajectory
from tools import r6_video_presentation as m
from tools import r6_video_trajectory as decoder
from tools.prepare_r6_video_trajectories import prepare
from tools.r6_recovery import file_pin, RecoveryError


def test_recorded_failure_and_reset_break(trajectory):
    _, _, write = trajectory
    decoded = decoder.decode(write())
    # Explicit presentation fixture with an additional mid-clip reset.
    for i, row in enumerate(decoded['frames']):
        if i == 1:
            row['reset_after_state'] = True
            row['collision'] = True
        if i >= 1:
            row['episode_counter_after_step'] += 1
    frames = m.frames(decoded)
    assert len(frames) == 200
    assert frames[1]['recorded']['collision'] and len(frames[1]['trail']) == 2
    assert frames[2]['trail_break'] and len(frames[2]['trail']) == 1
    assert frames[-1]['recorded']['reset_after_state']
    assert frames[-1]['recorded']['state_time_s'] == 20


@pytest.mark.parametrize('fault', ['pose', 'clock', 'quaternion', 'episode', 'nan', 'flag', 'partial'])
def test_bad_frame_presentation_rejected(trajectory, fault):
    _, _, write = trajectory
    decoded = decoder.decode(write())
    row = decoded['frames'][0]
    if fault == 'pose': row['position_local_m'][0] += 1
    if fault == 'clock': row['state_time_s'] = 0
    if fault == 'quaternion': row['quaternion_wxyz'] = [0, 0, 0, 0]
    if fault == 'episode': row['episode_counter_after_step'] = 2
    if fault == 'nan': row['geometry_distance_m'] = float('nan')
    if fault == 'flag': row['collision'] = 0
    if fault == 'partial': decoded['frames'].pop()
    with pytest.raises(RecoveryError): m.frames(decoded)


def mp4(path, count, *, blank_scene=False):
    import imageio.v2 as imageio
    import numpy as np
    frame = np.zeros((640, 960, 3), dtype=np.uint8)
    frame[:96, ::2, :] = 255  # Nonblank HUD must not disguise a blank scene.
    if not blank_scene:
        frame[96:490, ::4, :] = 255
    with imageio.get_writer(str(path), fps=10, codec='libx264', macro_block_size=1) as writer:
        for _ in range(count): writer.append_data(frame)
    return file_pin(path)


def test_actual_mp4_all_frames(tmp_path):
    pin = mp4(tmp_path/'valid.mp4', 3)
    assert m.verify_video(pin, 3)['all_frames_decoded']
    with pytest.raises(RecoveryError, match='FRAME_COUNT'): m.verify_video(pin, 4)
    with pytest.raises(RecoveryError, match='EXTRA_FRAMES'): m.verify_video(pin, 2)


def test_nonblank_hud_cannot_hide_blank_scene(tmp_path):
    pin = mp4(tmp_path/'blank.mp4', 2, blank_scene=True)
    with pytest.raises(RecoveryError, match='BLANK_SCENE'): m.verify_video(pin, 2)


def test_complete_raw_to_extraction_to_video_validation(trajectory, tmp_path):
    _, _, write = trajectory
    clip = write()
    selection = dict(schema_version='r6-video-selection/1', status='VIDEO_SOURCES_SELECTED_NOT_RENDERED',
        clips=[clip], total_frames=200, fps=10, physics_reintegrated=False, dispatch_authorized=False)
    selection_path = tmp_path/'selection.json'
    selection_path.write_text(json.dumps(selection))
    prepare(file_pin(selection_path), tmp_path/'extracted')
    extraction_pin = file_pin(tmp_path/'extracted/result.json')
    clips = m.load(extraction_pin)
    media = tmp_path/'media'; media.mkdir()
    telemetry = media/'clip.json'
    expected = dict(trajectory=clips[0]['trajectory'], frames=clips[0]['frames'])
    telemetry.write_text(json.dumps(expected))
    entry = dict(index=0, trajectory=clips[0]['trajectory'],
        video=mp4(media/'synthetic.mp4', 200), telemetry=file_pin(telemetry))
    manifest = dict(schema_version='r6-video-render/1', status='RENDERED_PENDING_CPU_VALIDATION',
        extraction=extraction_pin, renderer=file_pin(Path(m.__file__).with_name('render_r6_isaac_video.py')),
        scene_sources=[file_pin(__file__), file_pin(m.__file__)],
        videos=[entry], physics_steps=0, policy_inference_steps=0)
    path = media/'RUN_MANIFEST.json'; path.write_text(json.dumps(manifest))
    out = m.verify(file_pin(path), extraction_pin)
    assert not out['scientific_acceptance'] and out['videos'][0]['frames'] == 200
    bad = copy.deepcopy(expected); bad['frames'][-1]['recorded']['collision'] = False
    telemetry.write_text(json.dumps(bad))
    entry['telemetry'] = file_pin(telemetry); path.write_text(json.dumps(manifest))
    with pytest.raises(RecoveryError, match='TELEMETRY'): m.verify(file_pin(path), extraction_pin)
    # Rehashing a modified raw file is not enough: the original selection pins win.
    raw = clip['raw']['result.json']['path']
    Path(raw).write_text('{}')
    with pytest.raises(RecoveryError, match='PIN_DRIFT'): m.load(extraction_pin)
