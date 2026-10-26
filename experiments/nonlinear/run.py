"""Fresh protocol reproduction; frozen numerical sources remain unchanged."""
import argparse
import os
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['test', 'benchmark', 'search', 'main', 'evaluate', 'all'])
    parser.add_argument('--output', type=Path, default=Path('runs/nonlinear_reproduction'))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MUON_NONLINEAR_OUTPUT'] = str(output)
    os.environ['MPLBACKEND'] = 'Agg'
    protocol = here / 'protocol.json'
    saved = output / 'protocol.json'
    if saved.exists() and saved.read_bytes() != protocol.read_bytes():
        raise RuntimeError('Output protocol differs; choose a fresh output directory.')
    if not saved.exists():
        shutil.copyfile(protocol, saved)
    sys.path.insert(0, str(here / 'frozen'))
    import runner
    # Lock the run directory, including on Linux. No optimizer or RNG overrides.
    with (output / 'worker.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.execute_with_failure_status(args.action, lock)


if __name__ == '__main__':
    main()
