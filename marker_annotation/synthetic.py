"""Generate an explicitly fictional marker, never a substitute for real CAD."""
import json
from pathlib import Path
import cv2
import numpy as np
import yaml


def generate(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    corners = {}
    for group, points in [('plate', [(0,0,0),(.2,0,0),(.2,.15,0),(0,.15,0)]),
                           ('L-shape', [(.03,.03,.04),(.09,.03,.04),(.09,.06,.04),(.06,.06,.04),(.06,.10,.04),(.03,.10,.04)]),
                           ('cube', [(.13,.04,.05),(.18,.04,.05),(.18,.09,.05),(.13,.09,.05)])]:
        for i, xyz in enumerate(points): corners[f'{group[0].upper()}{i}'] = dict(group=group,xyz=xyz)
    edges=[]
    for group in ['plate','L-shape','cube']:
        keys=[k for k,c in corners.items() if c['group']==group]
        edges.extend([[a,b] for a,b in zip(keys, keys[1:]+keys[:1])])
    g=dict(name='SYNTHETIC fictional marker',units='m', axes='right / down / positive z away from camera at identity',corners=corners, edges=edges)
    (root/'geometry.json').write_text(json.dumps(g,indent=2))
    cal=dict(width=640,height=480,distortion_model='plumb_bob',k=[600,0,320,0,600,240,0,0,1],d=[0,0,0,0,0])
    xyz=np.array([c['xyz'] for c in corners.values()])
    for frame in range(3):
        rv=np.array([.15,-.22+frame*.08,.06]); tv=np.array([-.1,-.07,.65])
        uv=cv2.projectPoints(xyz,rv,tv,np.array(cal['k'],float).reshape(3,3),np.zeros(5))[0].reshape(-1,2)
        intensity=np.full((480,640),35,np.uint8); pts=dict(zip(corners,uv))
        for a,b in edges: cv2.line(intensity,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),180,2)
        for p in uv: cv2.circle(intensity,tuple(p.astype(int)),3,255,-1)
        cv2.putText(intensity,'SYNTHETIC EXAMPLE', (15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,230,1)
        cv2.imwrite(str(root/f'{frame}_intensity.png'),intensity)
        # No fabricated depth measurements: this demo tests optional missing depth.
        cv2.imwrite(str(root/f'{frame}_depth.png'),np.zeros((480,640),np.uint16))
        (root/f'{frame}_camera_params.yaml').write_text(yaml.safe_dump(cal))
    return g,cal,uv,rv,tv

if __name__=='__main__':
    import sys
    generate(sys.argv[1] if len(sys.argv)>1 else 'marker_annotation/example')
