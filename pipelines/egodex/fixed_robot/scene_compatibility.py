"""Conservative geometry identity for explicit physics-only scene revisions."""
import re
import xml.etree.ElementTree as ET
from .utils import read_json, sha256, digest

# Geometry, topology, contact masks and joint ranges intentionally remain fixed.
_PHYSICS = {
    'geom': {'mass','density','friction','solref','solimp','solmix'},
    'pair': {'friction','solref','solreffriction','solimp'},
    'joint': {'damping','frictionloss','armature','stiffness'},
    'inertial': {'mass','diaginertia','fullinertia'},
}


def historical_record(ctx, identifier, kind):
    if not re.fullmatch(kind + r'-[0-9a-f]{16}', identifier):
        raise ValueError('expected an existing ' + kind + ' record ID')
    record=read_json(ctx.path('.harness/records/'+identifier+'.json'))
    if record['id']!=identifier or record['kind']!=kind or record['input_signature']!=ctx._input_signature():
        raise ValueError('historical record identity/input mismatch')
    for path,signature in record['files'].items():
        if sha256(ctx.path(path))!=signature: raise ValueError('historical artifact changed: '+path)
    for dep,dep_id in record['dependencies'].items(): historical_record(ctx,dep_id,dep)
    return record


def layout_signature(xml_path):
    def node(element):
        attrs={k:v for k,v in element.attrib.items() if k not in _PHYSICS.get(element.tag,set())}
        if 'file' in attrs:
            from pathlib import Path
            path=Path(attrs['file'])
            if not path.is_absolute(): path=xml_path.parent/path
            attrs['file']=sha256(path)
        return [element.tag,attrs,[node(child) for child in element]]
    return digest(node(ET.parse(xml_path).getroot()))


def require_same_layout(ctx, old_scene, new_scene):
    if layout_signature(ctx.path(old_scene['scene_xml']))!=layout_signature(ctx.path(new_scene['scene_xml'])):
        raise ValueError('scene geometry/topology/layout changed; generate and select new placement candidates')
