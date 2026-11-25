"""Blender --background --python render_blender.py -- --input DIR [--camera ext1]."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Matrix, Vector

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--camera', default='ext1')
p.add_argument('--samples', type=int, default=48)
p.add_argument('--width', type=int, default=960)
p.add_argument('--height', type=int, default=540)
p.add_argument('--device', choices=['CPU','CUDA','OPTIX'], default='CUDA')
p.add_argument('--save-blend', action='store_true')
a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
start = time.monotonic()
meta = json.loads((a.input/'manifest.json').read_text())
with np.load(a.input / 'states.npz', allow_pickle=False) as archive:
    states = {name: archive[name] for name in archive.files}
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = a.samples
scene.cycles.use_denoising = True
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = .025
scene.cycles.max_bounces = 8
scene.render.use_persistent_data = True
if a.device != 'CPU':
    pref = bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type = a.device
    pref.get_devices()
    for d in pref.devices: d.use = d.type == a.device
    if not any(d.use for d in pref.devices): raise RuntimeError('Requested GPU device unavailable')
    scene.cycles.device = 'GPU'
scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.film_transparent = False
scene.view_settings.view_transform = 'AgX'
scene.view_settings.exposure = -.35
scene.world = bpy.data.worlds.new('neutral_world')
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.72,.76,.82,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .25
for name, loc, energy, size in [('key',(1,-2,3),180,2.2),('fill',(-2,0,2),80,2.8),('rim',(0,2,3),120,2)]:
    light=bpy.data.lights.new(name,'AREA'); light.energy=energy; light.shape='DISK'; light.size=size
    ob=bpy.data.objects.new(name,light); scene.collection.objects.link(ob); ob.location=loc
    ob.rotation_euler=(Vector((0,0,.5))-ob.location).to_track_quat('-Z','Y').to_euler()
objects=[]
for g in meta['geoms']:
    i, typ, s = g['id'], g['type'], g['size']
    if typ == 7:
        mesh=bpy.data.meshes.new(g['name'])
        mesh.from_pydata(states[f'vertices_{i}'].tolist(), [], states[f'faces_{i}'].tolist()); mesh.update()
        if f'face_uv_{i}' in states:
            uv=mesh.uv_layers.new(name='source_uv')
            for poly, values in zip(mesh.polygons,states[f'face_uv_{i}']):
                for loop, value in zip(poly.loop_indices,values): uv.data[loop].uv=value
        ob=bpy.data.objects.new(g['name'],mesh); scene.collection.objects.link(ob)
    elif typ == 6:
        bpy.ops.mesh.primitive_cube_add(size=2); ob=bpy.context.object; ob.scale=s
    elif typ in (2,4):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,radius=1)
        ob=bpy.context.object; ob.scale=([s[0]]*3 if typ==2 else s)
    elif typ in (3,5):
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=s[0],depth=2*s[1]); ob=bpy.context.object
        if typ==3:
            # Join sphere caps into one mesh so the exact geom transform moves the entire capsule.
            parts=[ob]
            for sign in (-1,1):
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,radius=s[0],location=(0,0,sign*s[1]))
                parts.append(bpy.context.object)
            bpy.ops.object.select_all(action='DESELECT')
            for part in parts: part.select_set(True)
            bpy.context.view_layer.objects.active=ob; bpy.ops.object.join()
    elif typ==0:
        bpy.ops.mesh.primitive_plane_add(size=2); ob=bpy.context.object
        ob.scale=(s[0] or 3,s[1] or 3,1)
    else:
        raise ValueError(f'Unsupported visible geom type {typ}: {g["name"]}')
    ob.name=g['name']
    material=bpy.data.materials.new(g['name']+'_material'); material.use_nodes=True
    bsdf=material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value=g['rgba']; bsdf.inputs['Roughness'].default_value=g.get('roughness',.48)
    bsdf.inputs['Metallic'].default_value=g.get('metallic',0)
    if g['rgba'][3] < 1.:
        bsdf.inputs['Alpha'].default_value=g['rgba'][3]
    if g.get('texture'):
        texture=material.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image=bpy.data.images.load(str((a.input/g['texture']).resolve()),check_existing=True)
        coord=material.node_tree.nodes.new('ShaderNodeTexCoord')
        mapping=material.node_tree.nodes.new('ShaderNodeVectorMath'); mapping.operation='MULTIPLY'
        repeat=g.get('texrepeat',[1,1]); mapping.inputs[1].default_value=(repeat[0],repeat[1],1)
        material.node_tree.links.new(coord.outputs['UV' if ob.data.uv_layers else 'Generated'],mapping.inputs[0])
        material.node_tree.links.new(mapping.outputs['Vector'],texture.inputs['Vector'])
        if not ob.data.uv_layers:
            texture.projection='BOX'
        multiply=material.node_tree.nodes.new('ShaderNodeMixRGB'); multiply.blend_type='MULTIPLY'
        multiply.inputs[0].default_value=1; multiply.inputs[2].default_value=g['rgba']
        material.node_tree.links.new(texture.outputs['Color'],multiply.inputs[1])
        material.node_tree.links.new(multiply.outputs[0],bsdf.inputs['Base Color'])
    if meta.get('appearance_profile')=='egodex-source-srgb':
        if not g.get('texture'):
            def linear(value):
                return value/12.92 if value<=.04045 else ((value+.055)/1.055)**2.4
            bsdf.inputs['Base Color'].default_value=(*[linear(v) for v in g['rgba'][:3]],g['rgba'][3])
        if typ==7:
            for poly in ob.data.polygons: poly.use_smooth=True
            edge=ob.modifiers.new('Preserve source hard edges','EDGE_SPLIT')
            edge.split_angle=.35
        if i in meta.get('cloth_geoms',[]):
            nodes=material.node_tree.nodes; links=material.node_tree.links
            coord=nodes.new('ShaderNodeTexCoord')
            weave=nodes.new('ShaderNodeTexNoise')
            weave.inputs['Scale'].default_value=330.
            weave.inputs['Detail'].default_value=2.
            links.new(coord.outputs['Generated'],weave.inputs['Vector'])
            bump=nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value=.20; bump.inputs['Distance'].default_value=.0006
            links.new(weave.outputs['Fac'],bump.inputs['Height'])
            links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
            bsdf.inputs['Roughness'].default_value=.82
    ob.data.materials.append(material)
    if typ in (2,3,4,5):
        for poly in ob.data.polygons: poly.use_smooth=True
    objects.append((ob,i,ob.scale.copy()))
camera_info=next((c for c in meta['cameras'] if c['name']==a.camera),None)
if camera_info is None: raise ValueError(f'Camera {a.camera} absent: {meta["cameras"]}')
camdata=bpy.data.cameras.new(a.camera); cam=bpy.data.objects.new(a.camera,camdata); scene.collection.objects.link(cam)
scene.camera=cam; camdata.sensor_fit='VERTICAL'; camdata.sensor_height=24
projection=camera_info.get('projection')
if projection:
    if projection['type']!='calibrated_pinhole': raise ValueError('Unsupported camera projection')
    # Use the same tested pinhole mapping as the vendored native modeler.
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'third_party/agenticSTAR/harness'))
    from rig.camera import apply_K
    source_w,source_h=projection['resolution']; sx=a.width/source_w; sy=a.height/source_h
    apply_K(camdata,projection['fx']*sx,projection['fy']*sy,
            (projection['cx']+.5)*sx-.5,(projection['cy']+.5)*sy-.5,a.width,a.height)
else:
    camdata.lens=12/math.tan(math.radians(camera_info['fovy'])/2)
directory=a.input/'frames'; directory.mkdir(exist_ok=True)
scene.render.fps=round(meta['fps'])
for frame in range(meta['frame_count']):
    for ob,i,scale in objects:
        matrix=Matrix(states['geom_rot'][frame,i].reshape(3,3)).to_4x4()
        matrix.translation=Vector(states['geom_pos'][frame,i]); ob.matrix_world=matrix; ob.scale=scale
    ci=camera_info['id']; matrix=Matrix(states['camera_rot'][frame,ci].reshape(3,3)).to_4x4()
    matrix.translation=Vector(states['camera_pos'][frame,ci]); cam.matrix_world=matrix
    bpy.context.view_layer.update()
    scene.render.filepath=str(directory/f'{frame:05d}.png')
    bpy.ops.render.render(write_still=True)
    if a.save_blend and frame==0: bpy.ops.wm.save_as_mainfile(filepath=str(a.input/'scene.blend'))
(a.input/'render_report.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,frames=meta['frame_count'],
    excluded_from_task_time=True,samples=a.samples,device=a.device,camera=a.camera,
    resolution=[a.width,a.height],preset='neutral-area-agx-v1'),indent=2))
