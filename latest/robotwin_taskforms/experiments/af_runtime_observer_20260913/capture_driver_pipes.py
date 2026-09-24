"""Read-only pipe observer for an already running, identity-checked driver.

Does not signal, restart, import, modify, or send input to the experiment.
Opening a new read end keeps the driver's stdout/stderr usable after SSH loss.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import time


def identity(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text()
    start_ticks = stat.rsplit(') ', 1)[1].split()[19]
    command = (proc / 'cmdline').read_bytes().split(b'\0')
    return start_ticks, [x.decode() for x in command if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--start-ticks', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    expected = identity(args.pid)
    if expected[0] != args.start_ticks:
        raise ValueError('Driver start time changed')
    if len(expected[1]) != 2 or expected[1][1] != 'launch_rim20_original_v3.py':
        raise ValueError('Not the expected running driver')
    args.out.mkdir(exist_ok=False)
    selector = selectors.DefaultSelector()
    streams = {}
    endpoints = {}
    try:
        for number, label, target in (
            (1, 'stdout', 'pipe:[7397719]'),
            (2, 'stderr', 'pipe:[7397720]'),
        ):
            path = Path('/proc') / str(args.pid) / 'fd' / str(number)
            if os.readlink(path) != target:
                raise ValueError('Driver pipe identity changed')
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            if identity(args.pid) != expected or os.readlink(path) != target:
                os.close(fd)
                raise ValueError('Driver identity changed while attaching')
            stream = (args.out / (label + '.log')).open('xb', buffering=0)
            streams[fd] = stream
            endpoints[label] = {'source_fd': str(path), 'pipe': target, 'access': 'read_only'}
            selector.register(fd, selectors.EVENT_READ)
        metadata = {
            'observer_pid': os.getpid(), 'driver_pid': args.pid,
            'driver_start_ticks': expected[0], 'driver_command': expected[1],
            'attached_unix_time': time.time(), 'endpoints': endpoints,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'experiment_signals_sent': False, 'experiment_inputs_written': False,
            'experiment_source_changed': False,
        }
        with (args.out / 'ATTACHED.json').open('x') as stream:
            json.dump(metadata, stream, indent=2)
        while selector.get_map():
            for key, _ in selector.select(timeout=1):
                try:
                    chunk = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                if chunk:
                    streams[key.fd].write(chunk)
                else:
                    selector.unregister(key.fd)
            try:
                alive = identity(args.pid) == expected
            except (FileNotFoundError, ProcessLookupError):
                alive = False
            if not alive:
                for fd in streams:
                    while True:
                        try:
                            chunk = os.read(fd, 65536)
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        streams[fd].write(chunk)
                break
        with (args.out / 'OBSERVER_EXIT.json').open('x') as stream:
            json.dump({'finished_unix_time': time.time(), 'reason': 'driver_identity_ended_or_pipes_closed'}, stream)
    finally:
        selector.close()
        for fd, stream in streams.items():
            stream.close()
            os.close(fd)


if __name__ == '__main__':
    main()
