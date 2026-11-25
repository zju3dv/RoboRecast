"""Every-physics-step ownership, contact continuity and object-relative slip."""
import numpy as np
import mujoco
from .robot_runtime import runtime_contract
from .contact_policy import declared_surface_pairs

class HandMonitor:
    def __init__(self, model, phase, acquisition_events=()):
        self.surface_pairs = {}
        self.permitted_pairs = set()
        self.phase=phase; self.metrics={}; self.reference={}; self.gap={}; self.first={}
        for side,item in phase.get('hand_objects',{}).items():
            if item['mode']=='idle': continue
            body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,item['object'])
            hand=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,runtime_contract(model)['arms'][side]['eef_body'])
            if body<0 or hand<0: raise ValueError('unknown hand/object binding')
            self.surface_pairs[side] = declared_surface_pairs(model, hand, body, item)
            self.permitted_pairs.update(self.surface_pairs[side])
            self.metrics[side]={'object':item['object'],'mode':item['mode'],'hold_policy':item.get('hold_policy','legacy_endpoint'),'steps':0,'pair_steps':0,'contact_steps':0,'max_contact_gap_sec':0.,'max_slip_m':0.,'max_rotation_slip_deg':0.,'final_pair':False,'final_contact':False,'min_height_m':None,'max_tilt_deg':0.}
            matching=[]
            if item['mode']=='grasp':
                for event in acquisition_events:
                    if not (phase['source_start']<=event['source_start']<=event['source_end']<=phase['source_end']):continue
                    for check in event.get('checks',[]):
                        if check.get('kind')=='grasp_acquired' and check.get('side')==side and check.get('body')==item['object']:
                            matching.append({'event':event['name'],'source_start':event['source_start'],'source_end':event['source_end'],'min_dwell_sec':check['min_dwell_sec']})
            self.metrics[side]['acquisition_policy']={'kind':'event_dwell' if matching else 'legacy_endpoint','events':matching}
            self.gap[side]=0.; self.reference[side]=None
    def step(self, model,data,contacts,check):
        for side,m in self.metrics.items():
            item=self.phase['hand_objects'][side]
            result=check(model,data,contacts,{'kind':'grasp','side':side,'body':item['object']})
            pair=result['passed']; contact=bool(result['measured']['fingers'])
            if self.surface_pairs[side]:
                contact = any(tuple(sorted((a,b))) in self.surface_pairs[side] for a,b,_,_ in contacts)
            m['steps']+=1;m['pair_steps']+=int(pair);m['contact_steps']+=int(contact);m['final_pair']=pair;m['final_contact']=contact
            self.gap[side]=0. if pair else self.gap[side]+model.opt.timestep
            m['max_contact_gap_sec']=max(m['max_contact_gap_sec'],self.gap[side])
            b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,item['object']);h=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,runtime_contract(model)['arms'][side]['eef_body'])
            height=float(data.xpos[b,2]);m['min_height_m']=height if m['min_height_m'] is None else min(m['min_height_m'],height)
            tilt=float(np.degrees(np.arccos(np.clip(data.xmat[b].reshape(3,3)[2,2],-1,1))))
            m['max_tilt_deg']=max(m['max_tilt_deg'],tilt)
            R=data.xmat[h].reshape(3,3); relative=R.T@(data.xpos[b]-data.xpos[h]); rotation=R.T@data.xmat[b].reshape(3,3)
            if self.reference[side] is None and (pair or item['mode']=='hold'):self.reference[side]=(relative.copy(),rotation.copy())
            if self.reference[side] is not None:
                p,r=self.reference[side];m['max_slip_m']=max(m['max_slip_m'],float(np.linalg.norm(relative-p)))
                angle=float(np.degrees(np.arccos(np.clip((np.trace(r.T@rotation)-1)/2,-1,1))))
                m['max_rotation_slip_deg']=max(m['max_rotation_slip_deg'],angle)
            reasons=[]
            if item['mode']=='hold':
                if m['max_contact_gap_sec']>float(item.get('max_contact_gap_sec',.15)):reasons.append('grasp_lost')
                if m['max_slip_m']>float(item.get('max_slip_m',.08)):reasons.append('grasp_slipped')
                if m['max_rotation_slip_deg']>float(item.get('max_rotation_slip_deg',180)):reasons.append('object_rotated')
            if item['mode'] in ('hold','grasp'):
                if 'max_tilt_deg' in item and m['max_tilt_deg']>float(item['max_tilt_deg']):reasons.append('object_tipped')
                if 'carry_height_m' in item and height<float(item['carry_height_m']):reasons.append('object_not_lifted')
            for reason in reasons:self.first.setdefault((side,reason),float(data.time))
    def failures(self):
        out=[{'kind':reason,'side':side,'object':self.metrics[side]['object'],'sim_time':time,'phase':self.phase['phase']} for (side,reason),time in self.first.items()]
        for side,m in self.metrics.items():
            mode=m['mode'];reason=None
            if mode=='grasp' and m['acquisition_policy']['kind']=='legacy_endpoint' and not m['final_pair']:reason='grasp_not_ready'
            if mode=='hold':
                if m['hold_policy']=='legacy_endpoint' and not m['final_pair']:reason='grasp_not_ready'
                if m['hold_policy']=='gap_slip' and not m['pair_steps']:reason='grasp_never_established'
            if mode=='release' and m['final_contact']:reason='release_not_complete'
            if mode=='contact' and m['contact_steps']/max(1,m['steps'])<float(self.phase['hand_objects'][side].get('min_contact_fraction',.25)):reason='contact_insufficient'
            if reason:out.append({'kind':reason,'side':side,'object':m['object'],'sim_time':self.phase['sim_end'],'phase':self.phase['phase']})
        return out
