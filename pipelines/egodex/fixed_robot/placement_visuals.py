"""State-neutral pose evidence, adapted from arm augmentation's shared-camera sheets."""
import itertools
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from .scene import render_image


def bounds_points(model, data):
    points=[]
    signs=np.array(list(itertools.product((-1,1), repeat=3)))
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE: continue
        # Compiled local AABBs include meshes; rotation transforms all eight corners.
        center=model.geom_aabb[i,:3]; half=model.geom_aabb[i,3:]
        points.extend((center+signs*half) @ data.geom_xmat[i].reshape(3,3).T + data.geom_xpos[i])
    return np.asarray(points)


def camera_spec(points):
    points=np.asarray(points);lo=points.min(axis=0);hi=points.max(axis=0)
    radius=max(float(np.linalg.norm(hi-lo)/2),.2)
    return {'lookat':((lo+hi)/2).tolist(),'distance':radius/np.sin(np.deg2rad(20))*1.12,
            'azimuth':135.,'elevation':-22.,'bounds_min':lo.tolist(),'bounds_max':hi.tolist()}


def camera(spec, azimuth=None):
    value=mujoco.MjvCamera();value.lookat[:]=spec['lookat'];value.distance=spec['distance']
    value.azimuth=spec['azimuth'] if azimuth is None else azimuth;value.elevation=spec['elevation']
    return value


def sheet(images,labels,path,columns):
    cells=[Image.open(p).convert('RGB') for p in images];w,h=cells[0].size
    result=Image.new('RGB',(w*columns,(h+38)*((len(cells)+columns-1)//columns)),'white');draw=ImageDraw.Draw(result)
    for i,(cell,label) in enumerate(zip(cells,labels)):
        x=i%columns*w;y=i//columns*(h+38);result.paste(cell,(x,y+38));draw.text((x+8,y+10),label,fill='black')
    result.save(path)


def detail_views(model,data,out):
    spec=camera_spec(bounds_points(model,data));renders={}
    for name,azimuth in [('global_left',75),('global_front',135),('global_right',195)]:
        path=out/(name+'.png');render_image(model,data,path,camera=camera(spec,azimuth));renders[name]=path
    for name in ('top','wrist_cam_left','wrist_cam_right'):
        path=out/(name+'.png');render_image(model,data,path,camera=name);renders[name]=path
    path=out/'selected_pose_2x3.png';sheet(list(renders.values()),list(renders),path,3)
    return renders,path,spec
