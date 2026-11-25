"""Small, inspectable failure evidence; never changes the production rollout."""
from pathlib import Path
import cv2
import mujoco
import numpy as np
from .utils import write_json
from .robot_runtime import runtime_contract


def classify_collision(model, geom1, geom2, held_objects=()):
    """Classify a geom pair. Expected grasp contacts still need phase semantics."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g)
             for g in (geom1, geom2)]
    bodies = [int(model.geom_bodyid[int(g)]) for g in (geom1, geom2)]
    def chain(body):
        result = []
        while body:
            result.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or '')
            body = int(model.body_parentid[body])
        return result
    ancestors = [chain(b) for b in bodies]
    robot = [runtime_contract(model)['base']['body'] in ns for ns in ancestors]
    payload = [bool(set(ns) & set(held_objects)) for ns in ancestors]
    finger_names={n for arm in runtime_contract(model)['arms'].values() for n in arm['gripper']['finger_bodies']}
    finger = [bool(finger_names & set(ns)) for ns in ancestors]
    if all(robot): category = 'robot_self_collision'
    elif any(robot) and any(payload):
        category = 'payload_finger_contact' if any(finger) else 'payload_robot_structure'
    elif any(robot): category = 'robot_environment_contact'
    elif any(payload): category = 'payload_environment_contact'
    else: category = 'scene_contact'
    return {'category': category, 'geoms': names}


def render_failure_evidence(ctx, model, states_path, diag, directory):
    """Render nearest recorded states at failure +/- 0.25s, with source RGB.

    Returns workspace-relative JPEG/JSON paths and a compact summary. Missing
    source evidence is explicit; simulated evidence remains useful for probes.
    """
    output = ctx.path(directory); output.mkdir(parents=True, exist_ok=True)
    with np.load(states_path) as z:
        states = {k: z[k].copy() for k in z.files}
    times = states['time']
    failure = diag.get('first_failure') or {}
    t = float(failure.get('sim_time', failure.get('time', times[-1])))
    indices = sorted(set(int(np.argmin(abs(times-x))) for x in (t-.25, t, t+.25)))
    width, height = 320, 180
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    camera_names = ('interaction_overview', 'top', 'wrist_cam_left', 'wrist_cam_right')
    source_error = None; source = None; cap = None
    try:
        from .inputs import load_reference
        _, source = load_reference(ctx.input_dir)
        cap = cv2.VideoCapture(str(Path(ctx.input_dir)/'source/rgb.mp4'))
        if not cap.isOpened(): raise ValueError('source video unavailable')
    except Exception as exc:
        source_error = str(exc)
    def labelled(rgb, label):
        rgb = np.ascontiguousarray(rgb.copy())
        cv2.rectangle(rgb, (0, 0), (width, 24), (15,15,15), -1)
        cv2.putText(rgb, label, (5,16), cv2.FONT_HERSHEY_SIMPLEX, .38, (245,245,245), 1)
        return rgb
    rows=[]; samples=[]
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for i in indices:
            data.qpos[:]=states['qpos'][i]; data.qvel[:]=states['qvel'][i]
            data.ctrl[:]=states['ctrl'][i]; data.time=float(times[i]); mujoco.mj_forward(model,data)
            panels=[]
            for name in camera_names:
                if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,name)>=0:
                    renderer.update_scene(data,camera=name); rgb=renderer.render()
                else: rgb=np.zeros((height,width,3),np.uint8)
                panels.append(labelled(rgb, f'{name} {times[i]:.2f}s'))
            source_time=float(states.get('source_time',times)[i]); frame=None
            rgb=np.zeros((height,width,3),np.uint8)
            if source is not None and cap is not None and cap.isOpened():
                si=int(np.argmin(abs(source['timestamps']-source_time)))
                frame=int(source['source_frames'][si]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
                ok,bgr=cap.read()
                if ok: rgb=cv2.cvtColor(cv2.resize(bgr,(width,height)),cv2.COLOR_BGR2RGB)
                else: source_error=f'cannot decode source frame {frame}'
            panels.append(labelled(rgb,f'source {source_time:.2f}s frame {frame}'))
            rows.append(np.concatenate(panels,axis=1))
            samples.append({'sim_time':float(times[i]),'source_time':source_time,'source_frame':frame})
    finally:
        renderer.close()
        if cap is not None:cap.release()
    sheet=output/'failure_keyframes.jpg'
    cv2.imwrite(str(sheet),cv2.cvtColor(np.concatenate(rows,axis=0),cv2.COLOR_RGB2BGR))
    summary={'first_failure':failure,'samples':samples,'source_evidence_error':source_error,
             'keyframes':str(sheet.relative_to(ctx.root)),
             'note':'Recorded-state replay; source RGB is temporal evidence, not contact truth.'}
    summary_path=output/'failure_summary.json';write_json(summary_path,summary)
    return {**summary,'summary_json':str(summary_path.relative_to(ctx.root))}
