"""Require controller-owned, completed trajectory extraction before render effects."""
import json
from tools.r6_recovery import RecoveryError, checked
from tools import r6_video_extraction as extraction


def validate(journal, parameters, purpose):
    journal.require_writer()
    if purpose != 'confirmation_delivery':
        raise RecoveryError('RENDER_JOB_PROTECTED_PURPOSE')
    state = journal.read()
    matches = []
    for key, event in state['events'].items():
        if event['identity']['role'] != extraction.ROLE or not event.get('receipt'):
            continue
        path=checked(event['receipt'])
        if path.stat().st_size>extraction.LOG_CAP:raise RecoveryError('RENDER_JOB_RECEIPT_CAP')
        receipt=json.loads(path.read_text())
        if receipt.get('outputs',{}).get('result.json') == parameters.get('extraction'):
            extraction.verified_receipt(journal, key)
            matches.append((key, event))
    if len(matches) != 1:
        raise RecoveryError('RENDER_JOB_EXACT_EXTRACTION_REQUIRED')
    key, event = matches[0]
    controller = state['events'].get(event['request']['controller_key'], {})
    cursor = controller.get('cursor', {})
    if (controller.get('identity', {}).get('role') != 'r6-controller'
            or cursor.get('video_extraction_key') != key
            or cursor.get('video_trajectories', {}).get('event_key') != key
            or cursor['video_trajectories'].get('status') != 'VIDEO_TRAJECTORIES_READY_NOT_RENDERED'
            or cursor['video_trajectories'].get('result') != parameters['extraction']
            or cursor['video_trajectories'].get('receipt') != event['receipt']):
        raise RecoveryError('RENDER_JOB_CONTROLLER_BINDING')
    return key
