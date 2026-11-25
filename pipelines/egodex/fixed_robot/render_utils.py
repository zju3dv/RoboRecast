"""Shared video encoding and panel labels; no evidence/physics imports."""
from pathlib import Path
import subprocess
import cv2
import numpy as np

class _Video:
    def __init__(self,path,width,height,fps):
        self.path=Path(path)
        self.process=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","-","-an","-c:v","libx264","-preset","fast","-crf","25","-pix_fmt","yuv420p","-movflags","+faststart",str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    def write(self,image):self.process.stdin.write(np.ascontiguousarray(image,dtype=np.uint8).tobytes())
    def close(self):
        self.process.stdin.close(); error=self.process.stderr.read(); code=self.process.wait()
        if code:raise RuntimeError(f"ffmpeg failed: {error.decode(errors='replace')}")


def _label(image,text):
    image=image.copy();cv2.rectangle(image,(0,0),(image.shape[1],26),(15,15,15),-1)
    cv2.putText(image,text,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
    return image
