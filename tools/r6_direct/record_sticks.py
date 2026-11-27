"""Record Linux joystick input independently of the Isaac simulation."""
import argparse
import json
import math
import os
from pathlib import Path
import select
import stat
import struct
import time

from tools.r6_direct.import_legacy import stop_paths


EVENT = struct.Struct('=IhBB')
JS_AXIS = 0x02
JS_INIT = 0x80


def velocity(raw_x, raw_y, forward_sign, lateral_sign, deadzone, max_speed):
    x = forward_sign * max(-1., raw_x/32767.)
    y = lateral_sign * max(-1., raw_y/32767.)
    length = math.hypot(x, y)
    if length <= deadzone:
        return [0., 0., 0.]
    speed = max_speed*(min(length, 1.)-deadzone)/(1.-deadzone)
    return [x*speed/length, y*speed/length, 0.]


def capture(args):
    if stop_paths():
        raise RuntimeError('STOP')
    mapping = {args.forward_axis: 0, args.lateral_axis: 1}
    descriptor = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if not stat.S_ISCHR(os.fstat(descriptor).st_mode):
            raise ValueError('a Linux joystick character device is required')
        raw = [0, 0]
        initialized = set()
        while select.select([descriptor], [], [], 0.)[0]:
            data = os.read(descriptor, EVENT.size)
            if len(data) != EVENT.size:
                raise OSError('incomplete joystick event')
            _, value, kind, axis = EVENT.unpack(data)
            if kind & ~JS_INIT == JS_AXIS and axis in mapping:
                raw[mapping[axis]] = value
                initialized.add(axis)
        if initialized != set(mapping):
            raise RuntimeError('joystick did not report both configured axes')

        def mapped():
            return velocity(*raw, args.forward_sign, args.lateral_sign, args.deadzone, args.max_speed)

        with args.output.open('x', encoding='utf-8', buffering=1) as handle:
            def record(kind, **data):
                print(json.dumps(dict(kind=kind, **data), allow_nan=False, separators=(',', ':')), file=handle)

            start = time.monotonic_ns()
            duration_ns = args.duration_s * 10**9
            record('header', schema_version='instinctrl-stick-recording/1', origin='physical_linux_joystick',
                   session=args.session, capture_start_unix_ns=time.time_ns(),
                   forward_axis=args.forward_axis, lateral_axis=args.lateral_axis,
                   forward_sign=args.forward_sign, lateral_sign=args.lateral_sign,
                   deadzone=args.deadzone, max_speed_m_s=args.max_speed, duration_ns=duration_ns)
            record('input', index=0, t_ns=0, device_ms=None, raw_axes=raw, velocity_w_m_s=mapped())
            recorded = 1
            while True:
                remaining = duration_ns - (time.monotonic_ns()-start)
                if remaining <= 0:
                    break
                if stop_paths():
                    raise RuntimeError('STOP')
                readable, _, _ = select.select([descriptor], [], [], min(1., remaining/10**9))
                if not readable:
                    continue
                data = os.read(descriptor, EVENT.size)
                received_ns = time.monotonic_ns()-start
                if len(data) != EVENT.size:
                    raise OSError('joystick disconnected or sent a partial event')
                if received_ns >= duration_ns:
                    break
                device_ms, value, kind, axis = EVENT.unpack(data)
                if kind == JS_AXIS and axis in mapping:
                    raw[mapping[axis]] = value
                    record('input', index=recorded, t_ns=received_ns, device_ms=device_ms,
                           raw_axes=raw, velocity_w_m_s=mapped())
                    recorded += 1
            record('complete', input_count=recorded, covered_duration_ns=duration_ns)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session', required=True, help='Non-identifying session label')
    parser.add_argument('--duration-s', type=int, required=True)
    parser.add_argument('--forward-axis', type=int, required=True)
    parser.add_argument('--lateral-axis', type=int, required=True)
    parser.add_argument('--forward-sign', type=int, choices=(-1, 1), required=True)
    parser.add_argument('--lateral-sign', type=int, choices=(-1, 1), required=True)
    parser.add_argument('--max-speed', type=float, default=.30)
    parser.add_argument('--deadzone', type=float, default=.05)
    args = parser.parse_args()
    if (not 1 <= args.duration_s <= 3600 or not args.session.strip()
            or args.forward_axis == args.lateral_axis
            or not all(0 <= axis <= 255 for axis in (args.forward_axis, args.lateral_axis))
            or not 0 < args.max_speed <= .30 or not 0 <= args.deadzone < 1):
        parser.error('duration, session, axis mapping or speed outside replay bounds')
    capture(args)


if __name__ == '__main__':
    main()
