import copy
import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsSimpleTextItem, QGraphicsPixmapItem
from marker_annotation.app import Window
from marker_annotation.synthetic import generate
from marker_annotation.viewer import Viewer


def setup_window(tmp_path):
    g,cal,uv,*_ = generate(tmp_path/'capture')
    w = Window(str(tmp_path/'capture'),str(tmp_path/'capture/geometry.json'),str(tmp_path/'session.json'))
    return w,uv


def add_clicks(w,uv):
    for i,p in enumerate(uv):
        w.corners.setCurrentRow(i); w.click(float(p[0]),float(p[1]))


def test_auto_fit_is_optional_debounced_saved_and_never_accepts(tmp_path):
    app = QApplication.instance() or QApplication([])
    w,uv = setup_window(tmp_path)
    assert not w.auto_estimate.isChecked()
    add_clicks(w,uv)
    assert w.h.value['pose'] is None
    w.auto_estimate.setChecked(True)
    assert w.auto_timer.isActive() and w.h.value['pose'] is None
    QTest.qWait(400)
    assert w.h.value['pose']['source'] == 'automatic'
    assert w.h.value['pose']['rmse_px'] < 1e-4
    assert not w.h.value['pose']['ambiguous']
    assert w.h.value['status'] == 'annotated'
    w.h.value['status'] = 'accepted'
    before = copy.deepcopy(w.h.value)
    w.estimate_automatically()
    assert w.h.value == before
    w.click(float(uv[-1][0]),float(uv[-1][1]))
    assert w.h.value['pose'] is None and w.h.value['status']=='annotated'
    w.estimate_automatically()
    assert w.h.value['pose'] is not None
    w.close()
    restored = Window(session=str(tmp_path/'session.json'))
    assert restored.auto_estimate.isChecked()
    restored.close()


def test_auto_waits_without_popups_and_hides_ambiguous_fit(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    w,uv = setup_window(tmp_path)
    monkeypatch.setattr(w,'error',lambda message: (_ for _ in ()).throw(AssertionError(message)))
    w.auto_estimate.setChecked(True)
    for i,p in enumerate(uv[:3]):
        w.corners.setCurrentRow(i); w.click(*p)
    w.estimate_automatically()
    assert w.h.value['pose'] is None and 'four' in w.auto_status.text()
    add_clicks(w,uv)
    candidate = w.fit_pose(); candidate['ambiguous'] = True
    monkeypatch.setattr(w,'fit_pose',lambda: copy.deepcopy(candidate))
    projections=[]
    original_show = w.viewer.show_image
    def capture(rgb,observations,selected,projected=None,edges=()):
        projections.append(projected)
        original_show(rgb,observations,selected,projected,edges)
    monkeypatch.setattr(w.viewer,'show_image',capture)
    w.estimate_automatically()
    assert w.h.value['pose']['ambiguous']
    assert w.h.value['status'] == 'annotated'
    assert projections[-1] is None
    assert 'competing solutions' in w.auto_status.text()
    w.auto_estimate.setChecked(False)
    assert not w.auto_timer.isActive()
    w.click(*uv[-1]); QTest.qWait(300)
    assert w.h.value['pose'] is None
    w.close()


def test_frame_switch_cannot_apply_pending_fit_to_another_frame(tmp_path):
    app = QApplication.instance() or QApplication([])
    w,uv = setup_window(tmp_path)
    w.auto_estimate.setChecked(True); add_clicks(w,uv)
    assert w.auto_timer.isActive()
    w.frames.setCurrentRow(1); QTest.qWait(400)
    assert not w.h.value['observations'] and w.h.value['pose'] is None
    w.close()


def test_annotations_and_model_are_translucent_image_is_opaque():
    app = QApplication.instance() or QApplication([])
    viewer = Viewer()
    viewer.show_image(np.zeros((100,100,3),np.uint8),
                      {'P0':{'xy':[20,20]},'P1':{'xy':[40,40]}},'P0',
                      {'P0':[21,20],'P1':[42,40]},[['P0','P1']])
    circles = [i for i in viewer.scene().items() if isinstance(i,QGraphicsEllipseItem)]
    labels = [i for i in viewer.scene().items() if isinstance(i,QGraphicsSimpleTextItem)]
    assert all(i.pen().color().alpha() <= 150 for i in circles)
    assert all(i.rect().width() <= 5 for i in circles)
    assert all(i.brush().color().alpha() <= 150 and i.font().pixelSize()==10 for i in labels)
    assert all(i.opacity()==1 for i in viewer.scene().items() if isinstance(i,QGraphicsPixmapItem))
    viewer.close()


def test_opacity_sliders_are_independent_and_preserve_review(tmp_path):
    from PySide6.QtWidgets import QGraphicsLineItem
    app = QApplication.instance() or QApplication([])
    w,uv = setup_window(tmp_path)
    add_clicks(w,uv)
    w.h.value.update(pose=w.fit_pose(),status='accepted')
    before = copy.deepcopy(w.h.value)
    w.marker_alpha.setValue(0); w.pose_alpha.setValue(100)
    labels = [i for i in w.viewer.scene().items() if isinstance(i,QGraphicsSimpleTextItem)]
    lines = [i for i in w.viewer.scene().items() if isinstance(i,QGraphicsLineItem)]
    assert labels and all(i.brush().color().alpha()==0 for i in labels)
    assert lines and any(i.pen().color().alpha()==255 for i in lines)
    w.marker_alpha.setValue(100); w.pose_alpha.setValue(0)
    assert all(i.brush().color().alpha()==255 for i in w.viewer.scene().items() if isinstance(i,QGraphicsSimpleTextItem))
    assert all(i.pen().color().alpha()==0 for i in w.viewer.scene().items() if isinstance(i,QGraphicsLineItem))
    assert w.h.value == before
    assert not w.auto_timer.isActive()
    w.close()
    restored = Window(session=str(tmp_path/'session.json'))
    assert restored.marker_alpha.value()==100 and restored.pose_alpha.value()==0
    assert restored.viewer.marker_opacity==1 and restored.viewer.overlay_opacity==0
    assert restored.h.value==before
    restored.close()
