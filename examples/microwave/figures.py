"""Make the README figures from three completed rollout directories."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
from door_figure import diagram
from roborecast.harness import load_candidate, read
from roborecast.keyframe import frame

HERE = Path(__file__).resolve().parent


def camera(center, distance, azimuth, elevation):
    view = mujoco.MjvCamera()
    view.type = mujoco.mjtCamera.mjCAMERA_FREE
    view.lookat[:] = center
    view.distance, view.azimuth, view.elevation = distance, azimuth, elevation
    return view


def montage(path, items):
    width, height, gap = 448, 252, 12
    sheet = Image.new("RGB", (width*len(items)+gap*(len(items)-1),height+57), "#f5f6f6")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 21)
    except OSError:
        font = ImageFont.load_default(size=21)
    for i, (picture, label) in enumerate(items):
        x = i*(width+gap)
        sheet.paste(ImageOps.fit(picture,(width,height)), (x,0))
        draw.text((x+12,height+13), label, font=font, fill="#182c32")
    sheet.save(path,quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for case in ["baseline","scene","grasp","out"]:
        parser.add_argument("--"+case,type=Path,required=True)
    parser.add_argument("--blocked-bundle", type=Path, required=True)
    parser.add_argument("--repaired-bundle", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, help="Write the first rim–trim contact and repaired clearance")
    a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    def shot(folder, edit, sec, view, coll=False, bundle=HERE):
        model,_=load_candidate(bundle,read(bundle/edit) if edit else {})
        with np.load(folder/"rollout.npz") as log:
            i=np.abs(log["timestamp_sec"]-sec).argmin()
            image=frame(model,log["qpos"][i],view,coll)
        return ImageEnhance.Brightness(image).enhance(1.2)
    wide=camera([.84,.06,.12],.92,45,-17)
    close=camera([.685,-.005,.105],.57,8,-13)
    def angle(folder):
        with np.load(folder/"rollout.npz") as log:
            return f'{np.degrees(log["joint:microwave__door_hinge"][-1]):.1f}°'
    ref=Image.open(HERE/"modeling/frames/000003.png")
    normal=shot(a.baseline,None,99,wide)
    montage(a.out/"opening.jpg",[(ref,"DROID demonstration"),(normal,"Simulated opening  ·  "+angle(a.baseline))])
    contact = diagram(a.blocked_bundle, a.repaired_bundle, a.scene/"rollout.npz", a.out)
    if a.contact_report:
        a.contact_report.write_text(json.dumps(contact, indent=2) + "\n")
    montage(a.out/"scene.jpg",[
        (shot(a.scene,None,99,wide,bundle=a.blocked_bundle),"Offset hinge axis  ·  "+angle(a.scene)),
        (shot(a.repair,None,99,wide,bundle=a.repaired_bundle),"Restored hinge axis  ·  "+angle(a.repair))])
    montage(a.out/"grasp.jpg",[(shot(a.grasp,"grasp-edit.json",7,close),"Offset grasp  ·  no contact"),
        (shot(a.baseline,None,7,close),"Restored handle contact"),(normal,"Door opens  ·  "+angle(a.baseline))])


if __name__ == "__main__":
    main()
