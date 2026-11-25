"""Independent task geometry, robot-free settling and source-view comparison."""
from pathlib import Path
import copy, shutil, uuid
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image
from .utils import write_json


def _directory(ctx, kind):
    path=ctx.path(f'artifacts/{kind}/{uuid.uuid4().hex}/placeholder').parent
    path.mkdir(parents=True,exist_ok=True)
    return path


def _relative(ctx,path): return str(Path(path).relative_to(ctx.root))


def register_scene_unit(ctx, request):
    role=request.get('role')
    if role not in ('task_object','support','required_obstacle'):
        raise ValueError('unit role must be task_object, support or required_obstacle; omit unrelated decor')
    reason=request.get('relevance_reason','').strip()
    if not reason: raise ValueError('state why this unit is necessary for manipulation/support/path clearance')
    source=ctx.path(request['fragment_xml_path'])
    root=ET.parse(source).getroot()
    if not request.get('evidence'): raise ValueError('unit requires visual/metric evidence')
    if root.find('.//include') is not None: raise ValueError('flatten XML includes before registration')
    if root.find('.//weld') is not None: raise ValueError('task welds are forbidden')
    if root.find('.//actuator') is not None: raise ValueError('task units cannot contain actuators')
    compiler = root.find('compiler')
    if any('euler' in e.attrib or ('range' in e.attrib and e.tag == 'joint' and e.get('type','hinge') == 'hinge') for e in root.iter()) and (compiler is None or compiler.get('angle') != 'radian'):
        raise ValueError('task angular values require compiler angle=radian')
    if any(not j.get('name') for j in [*root.iter('freejoint'), *root.iter('joint')] if j.get('joint') is None and j.get('joint1') is None):
        # Defaults are allowed to omit names; actual joints are checked during merge.
        for body in root.iter('body'):
            if any(not j.get('name') for j in [*body.findall('joint'),*body.findall('freejoint')]):
                raise ValueError('task joints must have names')
    unit=request['unit_id']
    if not unit or '/' in unit: raise ValueError('invalid unit_id')
    out=_directory(ctx,'unit')
    assets=[]
    compiler=root.find('compiler')
    for element in root.iter():
        if 'file' in element.attrib:
            filename=element.get('file')
            assetbase=source.parent
            if compiler is not None:
                subdir=compiler.get('meshdir' if element.tag=='mesh' else 'texturedir','')
                assetbase=assetbase/subdir
            asset=(assetbase/filename).resolve()
            dest=out/f'{len(assets)}_{asset.name}'
            shutil.copy2(asset,dest); element.set('file',str(dest)); assets.append(_relative(ctx,dest))
    if compiler is not None:
        compiler.attrib.pop('meshdir',None); compiler.attrib.pop('texturedir',None)
    xml=out/'unit.xml'; ET.ElementTree(root).write(xml,encoding='unicode')
    # MuJoCo validates inertias, dimensions, and limits before registration.
    mujoco.MjModel.from_xml_path(str(xml))
    try: registry=dict(ctx.latest('units')['payload']['units'])
    except ValueError: registry={}
    registry[unit]={'xml':_relative(ctx,xml),'assets':assets,'evidence':request['evidence'],'role':role,'relevance_reason':reason}
    files=[f for item in registry.values() for f in [item['xml'],*item['assets']]]
    return ctx.put('units',{'units':registry},files=files)


def remove_scene_unit(ctx,request):
    registry=dict(ctx.latest('units')['payload']['units'])
    unit=request['unit_id']
    if unit not in registry: raise ValueError('unknown unit_id')
    registry.pop(unit)
    files=[f for item in registry.values() for f in [item['xml'],*item['assets']]]
    return ctx.put('units',{'units':registry,'removed_unit':unit,'reason':request['reason']},files=files)


def merge_xml(target, source):
    for child in source:
        if child.tag in ('compiler','option','visual','statistic','size','keyframe'): continue
        section=target.find(child.tag)
        if section is None:
            target.append(copy.deepcopy(child))
        else:
            section.extend(copy.deepcopy(list(child)))


def merge_and_settle_scene(ctx,request):
    units=ctx.latest('units')['payload']['units']
    out=_directory(ctx,'scene')
    root=ET.Element('mujoco',model='human_task_scene')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep='.005',integrator='implicitfast')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='0 0 4')
    # Unit authors supply floor/support geometry in metric source-aligned world.
    for unit in units.values(): merge_xml(root,ET.parse(ctx.path(unit['xml'])).getroot())
    xml=out/'scene.xml'; ET.ElementTree(root).write(xml,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(xml)); data=mujoco.MjData(model)
    reused=None
    if request.get('reuse_scene_id'):
        from .scene_compatibility import historical_record, require_same_layout
        reused=historical_record(ctx,request['reuse_scene_id'],'scene')
        require_same_layout(ctx,reused['payload'],{'scene_xml':_relative(ctx,xml)})
        data.qpos[:]=np.load(ctx.path(reused['payload']['settled_state']))['qpos']
        mujoco.mj_forward(model,data)
    duration=float(request.get('settle_seconds',2))
    if not 0<duration<=20: raise ValueError('settle_seconds must be (0,20]')
    for _ in range(round(duration/model.opt.timestep)): mujoco.mj_step(model,data)
    speed=float(np.max(np.abs(data.qvel))) if model.nv else 0.
    penetration=max([float(-c.dist) for c in data.contact[:data.ncon]]+[0.])
    if not np.isfinite(data.qpos).all() or speed>.05 or penetration>.01:
        raise ValueError(f'robot-free settle unstable: max_speed={speed}, penetration={penetration}')
    state=out/'settled_state.npz'
    np.savez_compressed(state,qpos=data.qpos,qvel=np.zeros(model.nv),ctrl=data.ctrl)
    support_contacts=[]
    for contact_index,contact in enumerate(data.contact[:data.ncon]):
        force=np.zeros(6);mujoco.mj_contactForce(model,data,contact_index,force)
        support_contacts.append({'bodies':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in (contact.geom1,contact.geom2)],'geoms':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (contact.geom1,contact.geom2)],'normal_force':float(force[0]),'distance_m':float(contact.dist)})
    return ctx.put('scene',{'reused_scene_id':reused['id'] if reused else None,'unit_scope':{key:{'role':v.get('role'),'relevance_reason':v.get('relevance_reason')} for key,v in units.items()},'settled_contacts':support_contacts,'scene_xml':_relative(ctx,xml),'settled_state':_relative(ctx,state),'settle_diagnostics':{'duration_s':duration,'max_speed':speed,'max_penetration_m':penetration}},dependencies=['units'],files=[_relative(ctx,xml),_relative(ctx,state)])


def render_image(model,data,path,camera=None,intrinsics=None,image_size=None):
    with mujoco.Renderer(model,height=360,width=640) as renderer:
        if camera is None:
            camera=mujoco.MjvCamera(); camera.lookat[:]=[.3,0,.8]; camera.distance=2.8; camera.azimuth=135; camera.elevation=-25
        renderer.update_scene(data,camera=camera)
        if intrinsics is not None:
            width,height=image_size
            K=np.asarray(intrinsics)
            for glcamera in renderer.scene.camera:
                near=glcamera.frustum_near
                glcamera.frustum_top=float(K[1,2]/K[1,1]*near)
                glcamera.frustum_bottom=float(-(height-K[1,2])/K[1,1]*near)
                glcamera.frustum_center=float((width/2-K[0,2])/K[0,0]*near)
                glcamera.frustum_width=float(width/K[0,0]*near)
        Image.fromarray(renderer.render()).save(path)


def render_scene_comparison(ctx,request):
    scene=ctx.latest('scene')['payload']; out=_directory(ctx,'comparison')
    model=mujoco.MjModel.from_xml_path(str(ctx.path(scene['scene_xml'])));data=mujoco.MjData(model)
    data.qpos[:]=np.load(ctx.path(scene['settled_state']))['qpos'];mujoco.mj_forward(model,data)
    external=out/'external.png'; render_image(model,data,external)
    result={'external':_relative(ctx,external),'note':'External diagnostic view; source matching needs requested calibrated source frame.'}
    if 'source_frame' in request:
        from .inputs import load_reference
        import cv2
        from scipy.spatial.transform import Rotation
        meta,arrays=load_reference(ctx.input_dir); frame=int(request['source_frame'])
        transforms=np.asarray(arrays['T_world_camera']); K=np.asarray(arrays['K']); K=K[frame] if K.ndim==3 else K
        T=np.asarray(meta.get('T_sim_source_world',np.eye(4)))@transforms[frame]
        # Input camera uses OpenCV +Z forward/+Y down; MuJoCo uses -Z/+Y up.
        T = T.copy()
        T[:3,:3] = T[:3,:3] @ np.diag([1.,-1.,-1.])
        source_root=ET.parse(ctx.path(scene['scene_xml'])).getroot()
        cam=ET.SubElement(source_root.find('worldbody'),'camera',name='source_comparison',pos=' '.join(map(str,T[:3,3])),quat=' '.join(map(str,Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]])),fovy=str(np.degrees(2*np.arctan(meta.get('height',1080)/(2*K[1,1])))))
        source_xml=out/'source_camera.xml';ET.ElementTree(source_root).write(source_xml,encoding='unicode')
        sm=mujoco.MjModel.from_xml_path(str(source_xml));sd=mujoco.MjData(sm);sd.qpos[:]=data.qpos;mujoco.mj_forward(sm,sd)
        target=out/'source_view.png';render_image(sm,sd,target,'source_comparison',K,(meta['width'],meta['height']))
        cap=cv2.VideoCapture(str(Path(ctx.input_dir)/'source/rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,rgb=cap.read();cap.release()
        if not ok: raise ValueError('cannot read requested source frame')
        source_image=out/'source.png';cv2.imwrite(str(source_image),rgb)
        result.update(source=_relative(ctx,source_image),target=_relative(ctx,target),nominal_intrinsics=K.tolist(),note='Nominal K applied via an asymmetric camera frustum. Vision Pro RGB compositing can still differ from nominal intrinsics.')
    return result
