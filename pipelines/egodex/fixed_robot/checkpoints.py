"""Physics-step checkpoints with exact executable-prefix matching.

All integration state is retained, so replay can re-evaluate new checks and
produce complete prefix evidence without re-integrating unchanged controls.
"""
from pathlib import Path
import numpy as np
import mujoco
from .utils import digest, sha256, read_json, write_json

STATE = mujoco.mjtState.mjSTATE_INTEGRATION

def identity(ctx, placement_record, model):
    code = Path(__file__).parent
    # Record file hashes include the sealed scene assets; placement identity
    # deliberately invalidates all checkpoints after recommitting a placement.
    return digest({'input': ctx._input_signature(), 'placement': placement_record['id'],
                   'files': placement_record['files'], 'mujoco': mujoco.__version__,
                   'state_size': mujoco.mj_stateSize(model, STATE),
                   'code': {n: sha256(code/n) for n in ('rollout.py','checkpoints.py','monitoring.py','diagnostics.py','base_control.py','robot_runtime.py')}})

def load_prefix(ctx, key, model, times, controls, end, enabled=True, base_commands=None):
    pointer = ctx.path('.harness/physics_checkpoint.json')
    if not enabled or not pointer.exists(): return None, 0, 'disabled or no checkpoint'
    info = read_json(pointer)
    path = ctx.path(info['path'])
    if info['identity'] != key or not path.exists() or sha256(path) != info['sha256']:
        return None, 0, 'physics identity changed or checkpoint unavailable'
    with np.load(path) as archive: cached = {k: archive[k].copy() for k in archive.files}
    count = 0
    # State i was reached by control i-1. State at the first changed command is
    # safe to restore, but its current command must be replaced before stepping.
    for i, st in enumerate(cached['time']):
        if st > end+1e-9: break
        count = i+1
        command = np.array([np.interp(st,times,controls[:,j]) for j in range(model.nu)])
        if not np.array_equal(command, cached['control'][i]): break
        if base_commands is not None:
            base=base_command_at(st,times,base_commands)
            if 'base_control' not in cached or not np.array_equal(base,cached['base_control'][i]):break
    return cached, count, 'exact integration-state and control-prefix reuse'

def base_command_at(st,times,commands):
    if st>=times[-1]-1e-9:return np.zeros(2)
    return commands[max(0,int(np.searchsorted(times,st+1e-10,side='right')-1))].copy()

def save(ctx, key, path, times, states, controls, base_controls=None):
    extra={} if base_controls is None else {'base_control':base_controls}
    np.savez_compressed(path,time=times,integration_state=states,control=controls,**extra)
    write_json(ctx.path('.harness/physics_checkpoint.json'),
               {'identity':key,'path':str(path.relative_to(ctx.root)),'sha256':sha256(path)})
