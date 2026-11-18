import numpy as np
import cv2
from marker_annotation import state
from marker_annotation.pose import estimate, depth_check
from marker_annotation.synthetic import generate


def test_roundtrip_and_invalidation(tmp_path):
    r=state.record('capture/frame','tester'); r['status']='accepted'; r['pose']={'rmse_px':0}
    h=state.History(r); h.edit('P0',dict(xy=[10.25,20.75], visibility='visible',modality='intensity'))
    assert h.value['status']=='annotated' and h.value['pose'] is None
    path=tmp_path/'session.json'; state.save(path,dict(records=[h.value]))
    assert state.load(path)['records'][0]==h.value
    h.undo(); assert h.value['status']!='accepted'
    h.redo(); assert h.value['observations']['P0']['xy']==[10.25,20.75]


def test_pose(tmp_path):
    g,cal,uv,rv,tv=generate(tmp_path)
    obs={k:dict(xy=p.tolist(),visibility='visible',modality='intensity') for k,p in zip(g['corners'],uv)}
    result=estimate(g,obs,cal)
    assert result['rmse_px']<1e-5
    assert np.allclose(result['tvec_m'],tv,atol=1e-5)
    assert np.allclose(cv2.Rodrigues(np.array(result['rvec']))[0],cv2.Rodrigues(rv)[0],atol=1e-5)
    obs['P0']['visibility']='occluded'
    assert 'P0' in estimate(g,obs,cal)['excluded']


def test_planar_and_degenerate(tmp_path):
    import pytest
    g,cal,uv,*_=generate(tmp_path)
    obs={k:dict(xy=p.tolist(),visibility='visible',modality='intensity') for k,p in zip(g['corners'],uv) if k.startswith('P')}
    assert estimate(g,obs,cal)['planar']
    for o in obs.values(): o['xy']=[1,1]
    with pytest.raises(ValueError,match='Degenerate'): estimate(g,obs,cal)


def test_depth_edges():
    raw=np.full((20,20),1000,np.uint16)
    original=raw.copy()
    assert depth_check(raw,[10,10],.001)['reliable']
    raw[:,11:]=1500
    assert not depth_check(raw,[10,10],.001)['reliable']
    raw[10,10]=0
    assert not depth_check(raw,[10,10],.001)['reliable']


def test_view_coordinates():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPointF
    from marker_annotation.viewer import Viewer
    app=QApplication.instance() or QApplication([])
    v=Viewer(); v.resize(800,600)
    v.show_image(np.zeros((480,640,3),np.uint8),{},None); v.show(); app.processEvents()
    for scale in [.5,1,3]:
        v.resetTransform(); v.scale(scale,scale)
        v.horizontalScrollBar().setValue(30)
        p=QPointF(124,215)
        restored=v.mapToScene(v.mapFromScene(p))
        assert abs(restored.x()-p.x()) <= 1/scale
        assert abs(restored.y()-p.y()) <= 1/scale
    v.close()


def test_gui_session_restore(tmp_path):
    from PySide6.QtWidgets import QApplication
    from marker_annotation.app import Window
    app=QApplication.instance() or QApplication([])
    generate(tmp_path/'capture')
    session=tmp_path/'session.json'
    w=Window(str(tmp_path/'capture'),str(tmp_path/'capture/geometry.json'),str(session))
    w.click(123.25,234.5); w.persist(); w.close()
    restored=Window(session=str(session))
    assert restored.h.value['observations']['P0']['xy']==[123.25,234.5]
    restored.repeat()
    assert restored.h.value['observations']=={}
    assert len(restored.session['records'])==2
    restored.close()


def test_real_cad_profile(tmp_path):
    from pathlib import Path
    from marker_annotation.import_cad import convert
    from marker_annotation.app import Window
    from PySide6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    g=convert(Path(__file__).resolve().parents[2]/'data/helios2DepthMarker.ply')
    assert len(g['corners'])==14
    assert [k for k in g['corners'] if k.startswith('L')] == ['L00','L02','L05','L07','L09','L10']
    assert not any(c['cad_vertex_index'] in (5,6) for c in g['corners'].values())
    assert np.allclose(g['corners']['L00']['xyz'],[.13,.02,.06])
    assert g['corners']['P14']['group']=='plate'
    assert g['corners']['C28']['group']=='cube'
    assert len(g['edges'])<56*3
    w=Window(session=str(tmp_path/'session.json'),profile='pears')
    assert w.aligned.isChecked() and w.scale.text()=='0.001'
    assert w.g['name']=='helios2DepthMarker'
    w.close()
