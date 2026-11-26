"""Small, source-aligned visual and kinematic evidence packets (no scene mutation)."""
from pathlib import Path
import json
import cv2
import mujoco
import numpy as np
from mobile_input.loader import load_episode
from mobile_input.writer import write_json
from .semantic_trajectory import _arm


def crop_bounds(box, width, height, padding=.12):
    b = np.asarray(box, dtype=float)
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b < 0) or np.any(b > 1):
        raise ValueError('box_xyxy_normalized must be four finite values in [0,1]')
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError('crop box must have positive width and height')
    if not 0 <= padding <= .5:
        raise ValueError('padding must be in [0,.5]')
    delta = (b[2:] - b[:2]) * padding
    lo = np.floor(np.maximum(0, b[:2]-delta)*[width,height]).astype(int)
    hi = np.ceil(np.minimum(1, b[2:]+delta)*[width,height]).astype(int)
    return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]


def source_poses(episode, rows):
    model = mujoco.MjModel.from_xml_path(str(episode.runtime_scene))
    data = mujoco.MjData(model)
    with np.load(episode.initial_state) as a:
        initial = a['qpos'].copy()
    result=[]
    for row in rows:
        if not isinstance(row, int) or not 0 <= row < episode.rows:
            raise ValueError('row outside canonical source trajectory')
        data.qpos[:] = initial
        # Same planar state convention as the compiler and normalized inputs.
        data.qpos[:3] = episode.route_prior_xyyaw[row]
        hands={}
        for side, offset in [('left',0),('right',7)]:
            addresses, _, _ = _arm(model,side)
            data.qpos[addresses] = episode.arm_position[row,offset:offset+6]
        mujoco.mj_forward(model,data)
        for side, offset in [('left',0),('right',7)]:
            _, _, site = _arm(model,side)
            q=np.zeros(4);mujoco.mju_mat2Quat(q,data.site_xmat[site])
            hands[side]={'position_m':data.site_xpos[site].tolist(),
                         'quaternion_wxyz':q.tolist(),
                         'gripper_source_value':float(episode.arm_position[row,offset+6])}
        result.append({'row':row,'timestamp_sec':float(episode.timestamp_sec[row]),
                       'base_xyyaw':episode.route_prior_xyyaw[row].tolist(),'hands':hands})
    return result


def operation_reference(input_dir, store, request):
    episode=load_episode(input_dir)
    samples=request['samples']
    if not 1 <= len(samples) <= 12:
        raise ValueError('request 1..12 samples at a time')
    rows=list(dict.fromkeys(s['row'] for s in samples))
    poses=source_poses(episode,rows)
    folder=store.root/'preprocess/operation_references'
    folder.mkdir(parents=True,exist_ok=True)
    number=len(list(folder.glob('packet_*.json')))+1
    images=[]
    for index,s in enumerate(samples):
        role=s['camera'];row=s['row']
        if role not in episode.camera_paths:raise ValueError('unknown camera role')
        cap=cv2.VideoCapture(str(episode.camera_paths[role]))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,row);ok,frame=cap.read()
        finally:cap.release()
        if not ok:raise ValueError(f'cannot decode {role} row {row}')
        h,w=frame.shape[:2];stem=f'{number}_{index}_{role}_{row}'
        full=folder/f'{stem}_full.jpg';cv2.imwrite(str(full),frame)
        item={'camera':role,'row':row,'width':w,'height':h,'full_frame_path':str(full.relative_to(store.root))}
        if 'box_xyxy_normalized' in s:
            bounds=crop_bounds(s['box_xyxy_normalized'],w,h,s.get('padding',.12))
            x0,y0,x1,y1=bounds;crop=folder/f'{stem}_crop.jpg';cv2.imwrite(str(crop),frame[y0:y1,x0:x1])
            cv2.rectangle(frame,(x0,y0),(x1-1,y1-1),(0,255,255),2)
            boxed=folder/f'{stem}_boxed.jpg';cv2.imwrite(str(boxed),frame)
            item.update(crop_path=str(crop.relative_to(store.root)),boxed_frame_path=str(boxed.relative_to(store.root)),actual_box_xyxy_pixels=bounds)
        images.append(item)
    route={k:v for k,v in episode.episode_contract.items() if 'route' in k or 'base' in k}
    changes=[]
    for previous,current in zip(poses,poses[1:]):
        changes.append({'from_row':previous['row'],'to_row':current['row'],
            'hand_displacement_m':{side:float(np.linalg.norm(np.asarray(current['hands'][side]['position_m'])-
                previous['hands'][side]['position_m'])) for side in ('left','right')},
            'gripper_source_delta':{side:current['hands'][side]['gripper_source_value']-
                previous['hands'][side]['gripper_source_value'] for side in ('left','right')}})
    packet={'status':'PASS', 'sample_changes':changes,'coordinate_frame':'source route world; initial base reference, metres/radians; not reconstructed scene alignment',
            'route_provenance':route,'warning':'FK is a robot-model reference, not an object measurement. Crop pixels do not establish metric scale. Gripper value is source encoding, not normalized opening.',
            'poses':poses,'images':images}
    output=folder/f'packet_{number}.json';write_json(output,packet)
    return {**packet,'packet_path':str(output.relative_to(store.root))}
