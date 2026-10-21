"""Recorded R6 frame presentation and CPU validation, without simulation imports.

This module grants no execution permission. The caller owns render admission,
output reservation, process cleanup and the final independent acceptance.
"""
import json
import math
from pathlib import Path

from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_video_trajectory import bounded_json, vector


def require(ok, reason):
    if not ok:
        raise RecoveryError('VIDEO_PRESENTATION_' + reason)


def frames(trajectory):
    """Validate and present every recorded frame; break trails after resets."""
    t = trajectory
    require(t.get('schema_version') == 'r6-video-trajectory/1'
            and t.get('physics_reintegrated') is False
            and t.get('state_semantics') == 'RECORDED_PRE_RESET_POST_STEP_STATE'
            and t.get('command_semantics') == 'RECORDED_PRE_STEP_COMMAND'
            and t.get('reset_semantics') == 'DISPLAY_FAILURE_STATE_BEFORE_RESET_AND_BREAK_TRAIL_BEFORE_NEXT_FRAME', 'SEMANTICS')
    clip = t['clip']
    count = clip['frame_count']
    require(type(count) is int and count in (200, 330)
            and len(t['frames']) == count and clip['first_policy_step'] == 0
            and clip['last_policy_step'] == count - 1, 'FULL_TRAJECTORY')
    origin = vector(t['initial']['env_origin_w_m'], 3)
    history = []
    previous_reset = False
    episode = 0
    output = []
    for i, row in enumerate(t['frames']):
        root = vector(row['root_state_w'], 13)
        position = vector(row['position_local_m'], 3)
        quaternion = vector(row['quaternion_wxyz'], 4)
        vector(row['sensor_position_local_m'], 3)
        vector(row['command_w_m_s'], 3)
        vector(row['command_b_m_s'], 3)
        attitude = vector(row['control_attitude_wxyz'], 4)
        require(position == [root[k] - origin[k] for k in range(3)]
                and quaternion == root[3:7] and row['velocity_w_m_s'] == root[7:10]
                and abs(sum(x*x for x in quaternion) - 1) <= 1e-5
                and abs(sum(x*x for x in attitude) - 1) <= 1e-5, 'POSE')
        require(row['frame'] == i and row['policy_step'] == i
                and type(row['control_time_s']) in (int, float)
                and type(row['state_time_s']) in (int, float)
                and abs(row['control_time_s'] - i*.1) < 1e-9
                and abs(row['state_time_s'] - (i+1)*.1) < 1e-9, 'CLOCK')
        require(all(type(row[k]) is bool for k in ('reset_after_state', 'collision', 'height_violation')), 'FLAGS')
        episode += int(row['reset_after_state'])
        require(type(row['episode_counter_after_step']) is int
                and row['episode_counter_after_step'] == episode, 'EPISODE')
        distance = row['geometry_distance_m']
        require(type(distance) in (int, float) and math.isfinite(distance) and distance >= 0, 'DISTANCE')
        if previous_reset:
            history = []
        history.append(position)
        # Keep the failing post-step pose visible, then break before the next pose.
        output.append(dict(recorded=row, trail=list(history), trail_break=previous_reset))
        previous_reset = row['reset_after_state']
    return output


def load(extraction_pin):
    result = bounded_json(extraction_pin, 256*1024)
    require(result.get('status') == 'RECORDED_TRAJECTORIES_EXTRACTED_NOT_RENDERED'
            and result.get('fps') == 10 and result.get('physics_reintegrated') is False
            and result.get('dispatch_authorized') is False, 'EXTRACTION')
    selection = bounded_json(result['selection'], 256*1024)
    require(selection.get('status') == 'VIDEO_SOURCES_SELECTED_NOT_RENDERED'
            and selection.get('fps') == 10 and selection.get('physics_reintegrated') is False
            and 1 <= len(selection['clips']) <= 14
            and len(result['trajectories']) == len(selection['clips']), 'SELECTION')
    checked(result['extractor']); checked(result['decoder'])
    clips = []
    for i, entry in enumerate(result['trajectories']):
        decoded = bounded_json(entry['trajectory'], 16*1024**2)
        require(entry['index'] == i and decoded['clip'] == selection['clips'][i]
                and entry['frame_count'] == decoded['clip']['frame_count'], 'CLIP_BINDING')
        checked(decoded['decoder']); checked(decoded['audit_reader'])
        for pin in decoded['clip']['raw'].values():
            checked(pin)
        clips.append(dict(trajectory=entry['trajectory'], decoded=decoded, frames=frames(decoded)))
    require(sum(len(c['frames']) for c in clips) == result['total_frames'] == selection['total_frames'], 'TOTAL')
    return clips


def verify_video(video_pin, expected_frames):
    """Decode the complete MP4; check scene pixels separately from the HUD."""
    import cv2
    path = checked(video_pin)
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS) - 10) < .001, 'MP4_FPS')
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape == (640, 960, 3), 'MP4_SHAPE')
            require(float(frame[96:490, :, :].std()) > 5, 'BLANK_SCENE')
            count += 1
            require(count <= expected_frames, 'MP4_EXTRA_FRAMES')
        require(count == expected_frames, 'MP4_FRAME_COUNT')
    finally:
        cap.release()
    checked(video_pin)
    return dict(frames=count, fps=10, resolution=[960, 640], all_frames_decoded=True)


def verify(manifest_pin, extraction_pin):
    manifest = bounded_json(manifest_pin, 1024*1024)
    require(manifest.get('schema_version') == 'r6-video-render/1'
            and manifest.get('status') == 'RENDERED_PENDING_CPU_VALIDATION'
            and manifest.get('extraction') == extraction_pin
            and manifest.get('physics_steps') == 0
            and manifest.get('policy_inference_steps') == 0, 'MANIFEST')
    require(manifest['renderer'] == file_pin(Path(__file__).with_name('render_r6_isaac_video.py')), 'RENDERER')
    for pin in manifest['scene_sources']:
        checked(pin)
    require(len(manifest['scene_sources']) == 2, 'SCENE_SOURCES')
    clips = load(extraction_pin)
    require(len(manifest['videos']) == len(clips), 'VIDEO_COUNT')
    seen = set()
    output = []
    for i, (entry, clip) in enumerate(zip(manifest['videos'], clips)):
        require(entry['index'] == i and entry['trajectory'] == clip['trajectory'], 'VIDEO_BINDING')
        for key in ('video', 'telemetry'):
            path = checked(entry[key])
            require(path.parent == Path(manifest_pin['path']).parent and str(path) not in seen, 'OUTPUT_PATH')
            seen.add(str(path))
        telemetry = bounded_json(entry['telemetry'], 16*1024**2)
        require(telemetry == dict(trajectory=clip['trajectory'], frames=clip['frames']), 'TELEMETRY')
        decoded = verify_video(entry['video'], len(clip['frames']))
        output.append(dict(index=i, video=entry['video'], telemetry=entry['telemetry'], **decoded))
    checked(manifest_pin)
    return dict(status='VIDEOS_DECODED_AND_TELEMETRY_MATCHED_NOT_SCIENTIFIC_ACCEPTANCE',
                render_manifest=manifest_pin, extraction=extraction_pin, videos=output,
                validator=file_pin(__file__), physics_reintegrated=False, scientific_acceptance=False,
                limitation='Decoded pixels and matching telemetry do not independently prove pose-to-pixel fidelity; visual audit required.')
