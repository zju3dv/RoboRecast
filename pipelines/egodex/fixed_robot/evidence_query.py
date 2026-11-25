"""Deterministic extra recorded-state images/clips for reconstruction and review."""
from pathlib import Path
import numpy as np
from .inspection import _record
from .evidence import EvidenceRenderer, SETTINGS
from .render_utils import _Video
from .utils import digest, read_json, write_json, sha256


def render_execution_evidence(ctx, request):
    kind=request.get('kind','rollout')
    if kind not in ('rollout','stage_probe'):raise ValueError('kind must be rollout/stage_probe')
    record=_record(ctx,request['record_id']) if request.get('record_id') else ctx.latest(kind)
    if record['kind'] not in ('rollout','stage_probe'):raise ValueError('execution record required')
    placement=_record(ctx,record['dependencies']['placement'])
    key=digest({'record':record['id'],'files':record['files'],'request':request,'settings':SETTINGS})
    folder=ctx.path(Path('evidence_queries')/key[:20]);folder.mkdir(exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():
        cached=read_json(result_path)
        if all(ctx.path(p).exists() and sha256(ctx.path(p))==h for p,h in cached['hashes'].items()):
            return {**cached,'cache_hit':True}
    renderer=EvidenceRenderer(ctx,placement['payload'],ctx.path(record['payload']['states_npz']))
    try:
        times=renderer.states['time'];requested=request.get('times',[])
        if len(requested)>12 or not np.isfinite(requested).all():raise ValueError('at most 12 finite simulation times')
        if any(t<times[0] or t>times[-1] for t in requested):raise ValueError('time outside recorded execution')
        output_format = request.get('format', 'keyframes')
        if output_format not in ('keyframes', 'video'): raise ValueError('format must be keyframes/video')
        rows=sorted(set(int(np.argmin(abs(times-t))) for t in requested))
        interval = None
        if 'start' in request or 'end' in request:
            if 'start' not in request or 'end' not in request: raise ValueError('provide both start and end')
            start,end=float(request['start']),float(request['end'])
            if not np.isfinite([start,end]).all() or not times[0]<=start<end<=times[-1]:
                raise ValueError('interval must be within recording')
            interval = (start, end)
            if output_format == 'video' and end-start>8: raise ValueError('video interval must be <=8 seconds')
            if output_format == 'keyframes':
                count = int(request.get('frame_count', 8))
                if not 2 <= count <= 12: raise ValueError('frame_count must be 2..12')
                rows = sorted(set(rows + [int(np.argmin(abs(times-t))) for t in np.linspace(start, end, count)]))
                if len(rows)>12: raise ValueError('combined times and interval may select at most 12 frames')
        sheets=renderer.sheets(folder,[{'row':i, 'reasons':['requested time window' if interval else 'requested time']} for i in rows]) if rows else []
        clips=[]
        if output_format == 'video':
            if interval is None: raise ValueError('video requires start/end')
            start, end = interval
            path=folder/'clip.mp4';fps=SETTINGS['clip_fps'];writer=_Video(path,renderer.width*3,renderer.height*2,fps)
            mappings=[]
            try:
                for t in np.arange(start,end,1/fps):
                    i=int(np.argmin(abs(times-t))); writer.write(renderer.panel(i)); mappings.append(renderer.mapping(i))
            finally:writer.close()
            clips=[{'path':str(path.relative_to(ctx.root)),'start':start,'end':end,'fps':fps,'speed':1.0,'frames':mappings}]
        if not sheets and not clips:raise ValueError('provide times or start/end')
        result={'record_id':record['id'],'keyframes':sheets,'motion_clips':clips,
                'samples':[{**renderer.mapping(i), 'sheet':p} for i,p in zip(rows,sheets)],
                'camera_plan':renderer.camera_plan,
                'note':'Recorded-state evidence only; no simulation or verdict changes. Source image alignment is not required.'}
    finally:renderer.close()
    paths=sheets+[c['path'] for c in clips]
    result['hashes']={p:sha256(ctx.path(p)) for p in paths};write_json(result_path,result)
    return {**result,'cache_hit':False}
