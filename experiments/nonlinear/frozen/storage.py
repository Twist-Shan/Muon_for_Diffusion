"""Append-only, hash-verified commits for Windows/synchronized directories.

The public file is a convenience mirror. Readers use committed immutable versions,
so an interrupted mirror write or a reader holding the old file cannot lose state.
Legacy files without version history remain readable. No rename is required.
"""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import warnings


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _versions(path):
    path = Path(path)
    return path.parent / '.versions' / path.name


def resolve_artifact(path):
    """Return the latest fully committed, verified payload; never use partials."""
    path = Path(path)
    directory = _versions(path)
    markers = sorted(directory.glob('*.commit.json'), reverse=True) if directory.exists() else []
    for marker in markers:
        try:
            entry = json.loads(marker.read_text(encoding='utf-8'))
            name = entry['file']
            if Path(name).name != name or entry['logical_name'] != path.name:
                continue
            payload = directory / name
            if payload.stat().st_size == entry['size'] and _digest(payload) == entry['sha256']:
                return payload
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if markers:
        # Once journaled, a mirror is never an authority (it may be half-written).
        raise OSError('No valid committed version: ' + str(path))
    if path.is_file():
        return path
    raise FileNotFoundError(path)


def artifact_exists(path):
    try:
        resolve_artifact(path)
        return True
    except OSError:
        return False


def list_artifacts(directory, pattern):
    directory = Path(directory)
    names = {p.name for p in directory.glob(pattern) if p.is_file()}
    versions = directory / '.versions'
    if versions.exists():
        names.update(p.name for p in versions.iterdir()
                     if p.is_dir() and fnmatch.fnmatch(p.name, pattern))
    return [directory / name for name in sorted(names) if artifact_exists(directory / name)]


_warned_mirrors = set()


def _mirror(payload, path):
    """Best effort only; never overwrite an immutable payload through a hardlink."""
    try:
        if path.suffix == '.pt':
            if path.exists():
                return
            try:
                os.link(payload, path)
            except OSError:
                with path.open('xb') as dst, payload.open('rb') as src:
                    shutil.copyfileobj(src, dst)
        else:
            with path.open('wb') as dst, payload.open('rb') as src:
                shutil.copyfileobj(src, dst)
    except OSError as exc:
        if str(path) not in _warned_mirrors:
            warnings.warn('Committed artifact is safe; mirror unavailable: ' + str(path) + ': ' + repr(exc))
            _warned_mirrors.add(str(path))


def write_artifact(path, writer):
    """writer(new_path) must close the file. Commit after flush/fsync and hashing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = _versions(path)
    directory.mkdir(parents=True, exist_ok=True)
    # Retry only transient I/O failures, each time using a fresh name. Partial
    # files are retained for diagnosis and never participate in restoration.
    for attempt in range(4):
        token = f'{time.time_ns():020d}_{uuid.uuid4().hex}'
        payload = directory / (token + path.suffix)
        marker = directory / (token + '.commit.json')
        try:
            writer(payload)
            with payload.open('r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            entry = {'version': 1, 'logical_name': path.name, 'file': payload.name,
                     'size': payload.stat().st_size, 'sha256': _digest(payload)}
            with marker.open('x', encoding='utf-8', newline='\n') as f:
                json.dump(entry, f, sort_keys=True, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep((0.1, 0.3, 0.8)[attempt])
    _mirror(payload, path)
    return payload


def write_json(path, obj):
    data = json.dumps(obj, indent=2, allow_nan=False).encode('utf-8')
    def writer(payload):
        with payload.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    return write_artifact(path, writer)


def read_json(path):
    return json.loads(resolve_artifact(path).read_text(encoding='utf-8-sig'))
