"""Object-to-camera pose in metres; OpenCV camera axes right/down/forward."""
import cv2
import numpy as np

SETTINGS = dict(planar_rank_ratio=1e-5, ambiguity_delta_px=0.5,
                depth_patch_radius=2, depth_discontinuity_m=0.02)


def depth_check(raw, xy, scale):
    x, y = np.rint(xy).astype(int)
    if raw is None or x < 2 or y < 2 or y >= raw.shape[0]-2 or x >= raw.shape[1]-2:
        return dict(reliable=False, reason='depth missing or patch outside image')
    patch = raw[y-2:y+3, x-2:x+3].astype(float)*scale
    if not (np.isfinite(patch) & (patch > 0)).all():
        return dict(reliable=False, reason='invalid depth in 5×5 patch')
    if np.ptp(patch) > SETTINGS['depth_discontinuity_m']:
        return dict(reliable=False, reason='depth discontinuity in 5×5 patch')
    return dict(reliable=True, z_m=float(np.median(patch)))


def estimate(g, observations, cal, modality='intensity', aligned=False, depth=None, scale=None):
    if not cal:
        raise ValueError('Per-frame camera calibration is missing.')
    K = np.asarray(cal['k'], float).reshape(3, 3)
    d = np.asarray(cal.get('d', []), float)
    used, excluded = [], {}
    for key, o in observations.items():
        if key not in g['corners']:
            excluded[key] = 'unknown geometry corner'
        elif o['visibility'] != 'visible' or o.get('xy') is None:
            excluded[key] = o['visibility'] + ' / no usable click'
        elif o.get('modality', 'intensity') != modality and not aligned:
            excluded[key] = 'different image; alignment unconfirmed'
        else:
            used.append(key)
    if len(used) < 4:
        raise ValueError(f'Need at least four visible correspondences; {len(used)} usable. Excluded: {excluded}')
    xyz = np.array([g['corners'][k]['xyz'] for k in used], float)
    uv = np.array([observations[k]['xy'] for k in used], float)
    singular = np.linalg.svd(xyz-xyz.mean(0), compute_uv=False)
    if singular[1] < max(singular[0]*1e-5, 1e-10) or np.linalg.matrix_rank(uv-uv.mean(0)) < 2:
        raise ValueError('Degenerate correspondences: corners are collinear or coincident.')
    planar = singular[2] < singular[0]*SETTINGS['planar_rank_ratio']
    flag = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP
    result = cv2.solvePnPGeneric(xyz, uv, K, d, flags=flag)
    candidates = []
    for rv, tv in zip(result[1], result[2]):
        R = cv2.Rodrigues(rv)[0]
        camera = xyz @ R.T + tv.reshape(3)
        if (camera[:, 2] <= 0).any():
            continue
        projected = cv2.projectPoints(xyz, rv, tv, K, d)[0].reshape(-1, 2)
        residuals = np.linalg.norm(projected-uv, axis=1)
        checks = {}
        if scale is not None and (aligned or modality == 'depth'):
            for i, key in enumerate(used):
                check = depth_check(depth, uv[i], scale)
                if check['reliable']:
                    check['error_m'] = float(check['z_m']-camera[i, 2])
                checks[key] = check
        candidates.append(dict(rvec=rv.ravel().tolist(), tvec_m=tv.ravel().tolist(),
                               rmse_px=float(np.sqrt(np.mean(residuals**2))),
                               residuals_px=dict(zip(used, residuals.tolist())), depth=checks))
    if not candidates:
        raise ValueError('No solution with all used corners in front of the camera.')
    candidates.sort(key=lambda c: c['rmse_px'])
    best = candidates[0].copy()
    best.update(used=used, excluded=excluded, planar=bool(planar),
                ambiguous=len(candidates)>1 and candidates[1]['rmse_px']-best['rmse_px'] < 0.5,
                candidates=candidates, settings=SETTINGS, modality=modality,
                convention='X_camera = Rodrigues(rvec) @ X_marker + tvec_m')
    return best
