import hashlib
import json
import numpy as np

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def validate(contract):
    if set(contract) != {'I', 'c', 'G'} or not isinstance(contract['I'], list) or not contract['I']:
        raise ValueError('Contract needs nonempty I and fixed c, G predicate lists')
    ids = [event['id'] for event in contract['I']]
    if len(set(ids)) != len(ids) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError('Event IDs must be unique nonempty strings')
    for event in contract['I']:
        if not event.get('predicates') or not event.get('roles'):
            raise ValueError('Each event needs roles and predicates')
        if any(x not in ids or x == event['id'] for x in event.get('after', [])):
            raise ValueError('Unknown/self precedence reference')
    done = set()
    while len(done) < len(ids):
        ready = {e['id'] for e in contract['I'] if set(e.get('after', [])) <= done} - done
        if not ready:
            raise ValueError('Cyclic precedence')
        done |= ready
    for predicate in [p for e in contract['I'] for p in e['predicates']] + contract['c'] + contract['G']:
        if predicate['type'] not in {'contact', 'delta', 'final', 'initial'} or not predicate.get('key'):
            raise ValueError('Unsupported predicate')
        if predicate['type'] == 'contact' and (not np.isfinite(predicate.get('min_duration_sec', 0)) or predicate.get('min_duration_sec', 0) <= 0):
            raise ValueError('Contact requires a positive sustained duration')
        if predicate['type'] != 'contact' and not ({'min', 'max'} & set(predicate)):
            raise ValueError('State predicate needs min/max bound')
    digest(contract)
    return contract

def associate(waypoints, contract):
    ids = {e['id'] for e in contract['I']}
    intervals = {key: [] for key in ids}
    elapsed = 0.
    for index, point in enumerate(waypoints):
        refs = point.get('requirement_ids', [])
        if set(refs) - ids:
            raise ValueError('Unknown waypoint requirement ID')
        if index == 0:
            continue
        duration = float(point['duration_sec'])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Duration must be positive')
        for key in refs:
            intervals[key].append([elapsed, elapsed + duration])
        elapsed += duration
    for key, windows in intervals.items():
        merged = []
        for start, end in windows:
            if merged and abs(merged[-1][1] - start) < 1e-9:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        intervals[key] = merged
    return intervals

def _predicate(predicate, evidence, window):
    times = np.asarray(evidence.get('timestamp_sec', []))
    key = predicate['key']
    if key not in evidence or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        return {'status': 'UNKNOWN', 'reason': 'missing or invalid timeline/observable', 'key': key}
    values = np.asarray(evidence[key])
    if len(values) != len(times) or not np.isfinite(values).all():
        return {'status': 'UNKNOWN', 'reason': 'invalid observable', 'key': key}
    start, end = window
    tolerance = float(np.max(np.diff(times))) * 1.01
    if start < times[0] - tolerance or end > times[-1] + tolerance:
        return {'status': 'UNKNOWN', 'reason': 'unexecuted interval', 'key': key}
    mask = (times >= start - 1e-9) & (times <= end + 1e-9)
    t, value = times[mask], values[mask]
    if len(t) < 2:
        return {'status': 'UNKNOWN', 'reason': 'insufficient interval samples', 'key': key}
    if 'component' in predicate:
        value = value[:, int(predicate['component'])]
    kind = predicate['type']
    if kind == 'contact':
        if value.ndim != 1:
            return {'status': 'UNKNOWN', 'reason': 'contact must be scalar', 'key': key}
        active = value > 0
        longest = current = 0.
        for i in range(1, len(t)):
            current = current + t[i] - t[i-1] if active[i] and active[i-1] else 0.
            longest = max(longest, current)
        measured = longest
        passed = longest + 1e-9 >= predicate['min_duration_sec']
    else:
        measured = value[-1] - value[0] if kind == 'delta' else value[0] if kind == 'initial' else value[-1]
        passed = bool(np.all(measured >= np.asarray(predicate.get('min', -np.inf))) and np.all(measured <= np.asarray(predicate.get('max', np.inf))))
    return {'status': 'PASS' if passed else 'FAIL', 'key': key, 'value': np.asarray(measured).tolist()}

def evaluate(contract, intervals, evidence):
    validate(contract)
    rows = []
    for event in contract['I']:
        windows = intervals.get(event['id'], [])
        checks = [_predicate(p, evidence, window) for window in windows for p in event['predicates']]
        # Every linked segment must satisfy the event; contiguous segments merge at compilation.
        status = 'UNKNOWN' if not checks or any(p['status'] == 'UNKNOWN' for p in checks) else 'PASS' if all(p['status'] == 'PASS' for p in checks) else 'FAIL'
        for prior in event.get('after', []):
            before = intervals.get(prior, [])
            if not before or not windows:
                status = 'UNKNOWN'
            elif max(x[1] for x in before) > min(x[0] for x in windows) + 1e-9:
                checks.append({'status': 'FAIL', 'reason': 'precedence', 'after': prior})
                status = 'FAIL'
        rows.append({'id': event['id'], 'status': status, 'checks': checks})
    times = evidence.get('timestamp_sec', [])
    full = [float(times[0]), float(times[-1])] if len(times) else [0, 0]
    constraints = {name: [_predicate(p, evidence, full) for p in contract[name]] for name in ('c', 'G')}
    accepted = all(r['status'] == 'PASS' for r in rows) and all(p['status'] == 'PASS' for checks in constraints.values() for p in checks)
    failures = sorted((r for r in rows if r['status'] != 'PASS'), key=lambda r: min((w[0] for w in intervals.get(r['id'], [])), default=float('inf')))
    return {'status': 'PASS' if accepted else 'FAIL', 'contract_sha256': digest(contract), 'events': rows, 'constraints': constraints, 'earliest_unmet': failures[0]['id'] if failures else None}
