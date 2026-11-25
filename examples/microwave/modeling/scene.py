import bpy
from shapes import box, set_color, apply_boolean

SCALE = 0.48
REFERENCE_FRAME = '000000.png'
DOOR_PARTS = ['door_rim','door_black_border','door_window','door_handle','handle_upper_mount','handle_lower_mount']
JOINTS = [{'name':'door_hinge','type':'revolute','child':DOOR_PARTS,'origin':(-.488,-.35,0),'axis':(0,0,-1),'limit':(0,105)}]
FRAMES = {
 '000000.png': {'pose':{'quaternion':(0.6052225891608947, 0.7899487407267757, -0.06710873364543905, 0.07199321122209938),'translation':(-0.10950700074650539, 0.1764553486598757, 0.8514443036578764)},'joints':{'door_hinge':0}},
 '000001.png': {'joints':{'door_hinge':10}},
 '000002.png': {'joints':{'door_hinge':20}},
 '000003.png': {'joints':{'door_hinge':55}},
}

def build():
    dark=(.035,.038,.04); silver=(.48,.49,.48); light=(.65,.66,.62)
    def block(name,c,s,color,metal=0,bevel=0):
        ob=box(name,c,s); set_color(ob,color,roughness=.32 if metal else .48,metallic=metal)
        if bevel:
            mod=ob.modifiers.new('edge_softening','BEVEL'); mod.width=bevel; mod.segments=3
            bpy.context.view_layer.objects.active=ob; bpy.ops.object.modifier_apply(modifier=mod.name)
        return ob
    # Hollow fixed housing: complete closed wall solids and front-open cavity.
    housing=block('housing',(0,.012,0),(1,.654,.56),dark,.35)
    cutter=box('cavity_cut',(-.07,-.055,0),(.805,.63,.474))
    apply_boolean(housing,cutter)
    block('cavity_floor',(-.07,-.025,-.227),(.80,.565,.014),light)
    block('cavity_ceiling',(-.07,-.025,.227),(.80,.565,.014),light)
    block('cavity_left',(-.465,-.025,0),(.014,.565,.44),light)
    block('cavity_right',(.325,-.025,0),(.014,.565,.44),light)
    block('cavity_back',(-.07,.245,0),(.79,.014,.44),light)
    block('fixed_fascia',(.423,-.327,0),(.142,.025,.535),silver,.7,.008)
    block('keypad_back',(.423,-.343,-.012),(.114,.009,.463),(.012,.017,.015),0,.009)
    block('display',(.423,-.350,.181),(.10,.006,.058),(.006,.04,.024))
    for i in range(4):
        # Green seven-segment-like display strokes; small enough to read as display at scene scale.
        for dz in [-.012,.012]:
            block('display_%d_%s'%(i,str(dz)),(.386+i*.024,-.354,.181+dz),(.015,.004,.003),(.025,.21,.08))
    for row in range(8):
        for col in range(3):
            block('key_%d_%d'%(row,col),(.389+col*.034,-.352,.105-row*.037),(.025,.008,.024),(.34,.35,.32),0,.003)
    block('bottom_trim',(0,-.310,-.263),(.98,.024,.026),silver,.6)
    # Door has a closed silver ring, a closed black surround, and an opaque smoked window.
    rim=block('door_rim',(-.075,-.354,0),(.823,.043,.54),silver,.65,.006)
    apply_boolean(rim,box('rim_cut',(-.075,-.354,0),(.768,.09,.480)))
    border=block('door_black_border',(-.075,-.354,0),(.766,.039,.478),(.008,.010,.009))
    apply_boolean(border,box('window_cut',(-.098,-.354,.005),(.633,.09,.340)))
    block('door_window',(-.098,-.354,.005),(.632,.031,.339),(.027,.033,.032),.16,.004)
    block('handle_upper_mount',(.297,-.395,.205),(.051,.075,.037),silver,.75,.006)
    block('handle_lower_mount',(.297,-.395,-.205),(.051,.075,.037),silver,.75,.006)
    block('door_handle',(.297,-.443,0),(.060,.040,.444),silver,.75,.009)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=.265,depth=.012,location=(-.07,-.025,-.209))
    tray=bpy.context.object; tray.name='fixed_turntable';set_color(tray,(.40,.44,.41),roughness=.24)
    for x in [-.4,.4]:
        for y in [-.24,.24]: block('foot_%s_%s'%(x,y),(x,y,-.291),(.07,.07,.025),(.018,.018,.018))
