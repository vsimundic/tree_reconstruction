"""Dataset adapters. Unknown conventions remain unknown."""
from pathlib import Path
import json
import cv2
import numpy as np
import yaml


def discover(root, folder_filters=()):
    root = Path(root)
    names = set(folder_filters)
    return sorted(p for p in root.rglob('*_intensity.png')
                  if not names or names.intersection(p.parent.parts))


def read_frame(path):
    path = Path(path)
    stem = path.name.removesuffix('_intensity.png')
    intensity = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(path.with_name(stem + '_depth.png')), cv2.IMREAD_UNCHANGED)
    if intensity is None:
        raise ValueError(f'Cannot read {path}')
    calibration = path.with_name(stem + '_camera_params.yaml')
    cal = yaml.safe_load(calibration.read_text()) if calibration.exists() else None
    if cal and cal.get('distortion_model') not in ('plumb_bob', None):
        raise ValueError('Unsupported distortion model: ' + cal['distortion_model'])
    return intensity, depth, cal, str(calibration)


def _compact_helios_geometry(g):
    """Remove CAD junctions 5/6 and compact L IDs once, with an explicit schema."""
    if g.get('name') != 'helios2DepthMarker' or g.get('corner_id_schema') == 'compact-l-v2':
        return {}
    original = g['corners']
    kept = {k:c for k,c in original.items() if c.get('cad_vertex_index') not in (5,6)}
    retained_l_vertices = (0,1,2,3,4,7,8,9,10,11,12,13)
    mapping = {k:(f"L{retained_l_vertices.index(c['cad_vertex_index']):02d}"
                  if c['group']=='L-shape' else k) for k,c in kept.items()}
    mapping.update({k:None for k in original if k not in kept})
    # The first release already omitted L06 from geometry, but old records may retain it.
    mapping.setdefault('L05', None); mapping.setdefault('L06', None)
    edges = [[mapping[a],mapping[b]] for a,b in g.get('edges', [])
             if mapping.get(a) is not None and mapping.get(b) is not None]
    by_vertex = {c.get('cad_vertex_index'):mapping[k] for k,c in kept.items()}
    for a,b in [(4,13),(9,11)]:
        if a in by_vertex and b in by_vertex:
            pair = [by_vertex[a],by_vertex[b]]
            if not any(set(e)==set(pair) for e in edges): edges.append(pair)
    g.update(corners={mapping[k]:c for k,c in kept.items()},edges=edges,
             corner_id_schema='compact-l-v2', retired_cad_vertices=[5,6],
             corner_id_convention='L00..L11 in retained CAD vertex order; P/C retain original vertex indices')
    return mapping


def normalize_helios_geometry(g):
    """Keep only the 4 plate, 4 cube, and 6 L-shape top corners."""
    if g.get('name') != 'helios2DepthMarker' or g.get('corner_id_schema') == 'top-surfaces-v3':
        return {}
    old_keys = set(g['corners'])
    mapping = _compact_helios_geometry(g)
    top_vertices = {0,2,7,9,11,12,14,16,17,24,26,27,28,29}
    kept = {k:c for k,c in g['corners'].items() if c.get('cad_vertex_index') in top_vertices}
    # Compose legacy renumbering with the top-surface restriction, using CAD identity.
    mapping = {k:(mapping.get(k,k) if mapping.get(k,k) in kept else None)
               for k in old_keys | set(mapping)}
    g.update(corners=kept,
             edges=[e for e in g.get('edges',[]) if all(k in kept for k in e)],
             corner_id_schema='top-surfaces-v3',
             retired_cad_vertices=sorted(set(range(30))-top_vertices),
             corner_id_convention='Top surfaces only; retain compact-l-v2 IDs and original CAD vertex metadata')
    return mapping


def migrate_helios_records(records, mapping):
    """Remap observations and fit diagnostics without reassigning physical points."""
    if not mapping: return
    def rename(values):
        return {mapping.get(k,k):v for k,v in values.items() if mapping.get(k,k) is not None}
    def rename_pose(pose):
        pose['used'] = [mapping.get(k,k) for k in pose.get('used',[]) if mapping.get(k,k) is not None]
        for field in ('excluded','residuals_px','depth'):
            if field in pose: pose[field] = rename(pose[field])
        for candidate in pose.get('candidates',[]): rename_pose(candidate)
    for record in records:
        observations = record['observations']
        removed = {k:v for k,v in observations.items() if mapping.get(k,k) is None}
        if removed: record.setdefault('retired_observations',{}).update(removed)
        record['observations'] = rename(observations)
        pose = record.get('pose')
        if removed or (pose and any(mapping.get(k,k) is None for k in pose.get('used',[]))):
            if pose:
                record.setdefault('superseded_poses',[]).append(pose)
            record['pose'] = None
            if record['status']=='accepted': record['status']='annotated'
        elif pose:
            rename_pose(pose)


def geometry(path):
    g = json.loads(Path(path).read_text())
    normalize_helios_geometry(g)
    if g['units'] != 'm':
        raise ValueError('Geometry coordinates must be in metres (units: m).')
    for key, corner in g['corners'].items():
        if len(corner['xyz']) != 3 or not np.isfinite(corner['xyz']).all():
            raise ValueError(f'Invalid corner {key}')
    for a, b in g.get('edges', []):
        if a not in g['corners'] or b not in g['corners']:
            raise ValueError('Edge refers to unknown corner')
    return g


def display_image(raw, depth=False):
    if raw is None:
        raise ValueError('No depth image for this frame')
    if raw.ndim == 3:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    valid = np.isfinite(raw) & (raw > 0)
    normalized = np.zeros(raw.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(raw[valid], [2, 98])
        normalized[valid] = (np.clip((raw[valid].astype(float)-lo)/max(hi-lo, 1), 0, 1)*255).astype(np.uint8)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    if depth:
        rgb[~valid] = [110, 65, 85]
    return np.ascontiguousarray(rgb)
