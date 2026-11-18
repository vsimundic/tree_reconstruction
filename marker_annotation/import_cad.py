"""Extract named CAD vertices and crease edges from the supplied Helios mesh.

Original PLY vertex indices are retained as metadata; this is a marker-specific adapter.
"""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from .data import normalize_helios_geometry


def convert(path):
    blob = Path(path).read_bytes()
    header, body = blob.split(b'end_header\n', 1)
    expected = [b'format binary_little_endian 1.0', b'element vertex 30', b'element face 56',
                b'property float x', b'property float y', b'property float z',
                b'property list uchar int vertex_indices']
    if not all(line in header.splitlines() for line in expected):
        raise ValueError('This adapter expects the supplied 30-vertex Helios marker PLY.')
    xyz = np.array([struct.unpack_from('<fff', body, i*12) for i in range(30)])*.001
    faces=[]; offset=360
    for _ in range(56):
        red, green, blue, alpha, n = struct.unpack_from('<BBBBB', body, offset); offset+=5
        face=struct.unpack_from('<'+'i'*n, body, offset); offset+=4*n
        if n!=3: raise ValueError('Expected triangular CAD faces')
        faces.append(face)
    neighbors={}
    for a,b,c in faces:
        normal=np.cross(xyz[b]-xyz[a],xyz[c]-xyz[a]); normal/=np.linalg.norm(normal)
        for edge in [(a,b),(b,c),(c,a)]: neighbors.setdefault(tuple(sorted(edge)),[]).append(normal)
    plate={14,15,16,17,18,19,22,24}
    groups={i:('plate' if i in plate else 'L-shape' if i<14 else 'cube') for i in range(30)}
    ids={i:('P' if groups[i]=='plate' else 'L' if groups[i]=='L-shape' else 'C')+f'{i:02d}' for i in range(30)}
    corners={ids[i]:dict(group=groups[i], xyz=p.tolist(), cad_vertex_index=i) for i,p in enumerate(xyz)}
    edges=[[ids[a],ids[b]] for (a,b),normals in neighbors.items()
           if len(normals)!=2 or abs(float(normals[0]@normals[1]))<.99999]
    result = dict(name='helios2DepthMarker', units='m',
                axes='Original CAD origin and axes retained: plate spans +X/+Y, thickness 0..5 mm; relief extends +Z to 60 mm. No recentering.',
                source=dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(blob).hexdigest(), units_to_m=.001),
                corners=corners,edges=edges)
    normalize_helios_geometry(result)
    return result

if __name__=='__main__':
    import sys
    Path(sys.argv[2]).write_text(json.dumps(convert(sys.argv[1]),indent=2))
