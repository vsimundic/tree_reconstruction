import json
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from marker_annotation.diagram import MarkerDiagram
from marker_annotation.app import Window


def test_all_fourteen_corners_share_one_view_without_label_overlap():
    app = QApplication.instance() or QApplication([])
    geometry = json.loads((Path(__file__).resolve().parents[1]/'helios_corners.json').read_text())
    diagram = MarkerDiagram(); diagram.set_geometry(geometry)
    diagram.resize(330,370); diagram.show(); app.processEvents()
    diagram.plan.layout_corners()
    assert len(diagram.plan.labels) == 14
    assert set(diagram.plan.labels) == set(geometry['corners'])
    labels = list(diagram.plan.labels.values())
    for i,a in enumerate(labels):
        assert diagram.plan.rect().contains(a.toAlignedRect())
        for b in labels[i+1:]: assert not a.intersects(b)
    diagram.close()


def test_diagram_and_list_select_across_heights_without_switching(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = Window(profile='pears', session=str(tmp_path/'annotations.json'))
    window.show(); app.processEvents()
    window.corners.setCurrentRow(window.keys.index('P14'))
    assert window.diagram.plan.selected == 'P14'
    assert 'Z = 5 mm' in window.diagram.detail.text()
    assert len(window.diagram.plan.labels) == 14
    window.diagram.plan.layout_corners()
    label = window.diagram.plan.labels['C28']
    QTest.mouseClick(window.diagram.plan, Qt.LeftButton, pos=label.center().toPoint())
    assert window.selected() == 'C28'
    assert 'Z = 60 mm' in window.diagram.detail.text()
    window.corners.setCurrentRow(window.keys.index('P14'))
    assert 'Plate top' in window.diagram.detail.text()
    window.close()


def test_rotation_mirror_preserve_ids_and_click_targets():
    import copy
    app = QApplication.instance() or QApplication([])
    geometry = json.loads((Path(__file__).resolve().parents[1]/'helios_corners.json').read_text())
    original = copy.deepcopy(geometry)
    diagram = MarkerDiagram(); diagram.set_geometry(geometry)
    diagram.set_selection('L00', {})
    diagram.resize(380,430); diagram.show(); app.processEvents()
    selected = []; diagram.corner_selected.connect(selected.append)
    for angle in [0,90,180,270,33]:
        for mirrored in [False,True]:
            diagram.set_orientation(angle,mirrored)
            assert diagram.plan.selected == 'L00'
            diagram.plan.layout_corners()
            labels = list(diagram.plan.labels.values())
            for i,a in enumerate(labels):
                for b in labels[i+1:]: assert not a.intersects(b)
            target = diagram.plan.labels['C28'].center().toPoint()
            QTest.mouseClick(diagram.plan,Qt.LeftButton,pos=target)
            assert selected[-1] == 'C28'
    diagram.set_orientation(90,False)
    x,y = diagram.plan.orient(1,0)
    assert abs(x)<1e-9 and abs(y-1)<1e-9
    diagram.set_orientation(0,True)
    assert diagram.plan.orient(1,0) == (-1,0)
    QTest.mouseClick(diagram.reset,Qt.LeftButton)
    assert diagram.rotation.value() == 0 and not diagram.mirror.isChecked()
    assert geometry == original
    diagram.close()


def test_orientation_saved_without_invalidating_review(tmp_path):
    import copy
    from marker_annotation.synthetic import generate
    from marker_annotation.pose import estimate
    app = QApplication.instance() or QApplication([])
    g,cal,uv,*_ = generate(tmp_path/'capture')
    session = tmp_path/'session.json'
    window = Window(str(tmp_path/'capture'),str(tmp_path/'capture/geometry.json'),str(session))
    obs = {k:dict(xy=p.tolist(),visibility='visible',modality='intensity') for k,p in zip(g['corners'],uv)}
    window.h.value.update(observations=obs,pose=estimate(g,obs,cal),status='accepted')
    before = copy.deepcopy(window.h.value)
    window.diagram.set_orientation(135,True)
    assert window.h.value == before
    window.close()
    restored = Window(session=str(session))
    assert restored.diagram.rotation.value() == 135 and restored.diagram.mirror.isChecked()
    assert restored.h.value == before
    restored.close()


def test_legacy_l06_is_archived_on_restore(tmp_path):
    from marker_annotation import state
    from marker_annotation.synthetic import generate
    app = QApplication.instance() or QApplication([])
    generate(tmp_path/'capture')
    g = json.loads((Path(__file__).resolve().parents[1]/'helios_corners.json').read_text())
    g.pop('corner_id_schema')
    g['corners'] = {('L'+f"{c['cad_vertex_index']:02d}" if c['group']=='L-shape' else k):c for k,c in g['corners'].items()}
    g['edges'] = []
    g['corners']['L06'] = dict(group='L-shape',xyz=[.02,.05,.005],cad_vertex_index=6)
    g['edges'].append(['L04','L06'])
    record = state.record(str((tmp_path/'capture/0_intensity.png').resolve()))
    observation = dict(xy=[100,200],visibility='visible',modality='intensity')
    record.update(observations={'L06': observation},status='accepted',pose={'used':['L06']})
    session = tmp_path/'legacy.json'
    state.save(session,dict(version=1,capture_root=str(tmp_path/'capture'),geometry={'definition':g},records=[record],conventions={}))
    window = Window(session=str(session))
    assert 'L06' not in window.g['corners']
    assert 'L06' not in window.h.value['observations']
    assert window.h.value['retired_observations']['L06'] == observation
    assert window.h.value['pose'] is None and window.h.value['status'] == 'annotated'
    assert window.h.value['superseded_poses'] == [{'used':['L06']}]
    window.close()


def test_top_surface_migration_preserves_retained_and_archives_lower_corners():
    import copy
    from marker_annotation import data, state
    g = json.loads((Path(__file__).resolve().parents[1]/'helios_corners.json').read_text())
    g['corner_id_schema'] = 'compact-l-v2'
    g['corners']['L06'] = dict(group='L-shape',xyz=[.05,.05,.005],cad_vertex_index=8)
    g['corners']['P15'] = dict(group='plate',xyz=[0,0,0],cad_vertex_index=15)
    retained = state.record('frame')
    retained.update(observations={'L05':{'xy':[10,20]},'C28':{'xy':[30,40]}},
                    status='accepted',pose=dict(used=['L05','C28'],residuals_px={'L05':.2}))
    lower = state.record('frame2')
    lower.update(observations={'L06':{'xy':[99,88]},'L05':{'xy':[55,66]}},status='accepted',pose={'used':['L06','L05']})
    mapping = data.normalize_helios_geometry(g)
    data.migrate_helios_records([retained,lower],mapping)
    assert g['corners']['L05']['cad_vertex_index'] == 7
    assert len(g['corners']) == 14
    assert retained['observations'] == {'L05':{'xy':[10,20]},'C28':{'xy':[30,40]}}
    assert retained['pose']['used'] == ['L05','C28'] and retained['status']=='accepted'
    assert lower['observations'] == {'L05':{'xy':[55,66]}}
    assert lower['retired_observations']['L06'] == {'xy':[99,88]}
    assert lower['pose'] is None and lower['status']=='annotated'
    before = copy.deepcopy([g,retained,lower])
    data.migrate_helios_records([retained,lower],data.normalize_helios_geometry(g))
    assert [g,retained,lower] == before


def test_top_surface_counts_heights_and_closed_outlines():
    from collections import Counter
    g = json.loads((Path(__file__).resolve().parents[1]/'helios_corners.json').read_text())
    assert Counter(c['group'] for c in g['corners'].values()) == {'plate':4,'cube':4,'L-shape':6}
    for c in g['corners'].values():
        assert c['xyz'][2] == (.005 if c['group']=='plate' else .06)
    degree = Counter(k for edge in g['edges'] for k in edge)
    assert set(degree) == set(g['corners'])
    assert all(count==2 for count in degree.values())
