from pathlib import Path
from PySide6.QtWidgets import QApplication, QDialog
from marker_annotation import data, state
from marker_annotation.startup import StartDialog
from marker_annotation.app import Window
from marker_annotation.synthetic import generate


def qt_app():
    return QApplication.instance() or QApplication([])


def test_folder_scope(tmp_path):
    marker = tmp_path/'A'/'capture'/'marker_captures'/'depth'
    other = tmp_path/'A'/'capture'/'tree_captures'/'depth'
    generate(marker); generate(other)
    assert len(data.discover(tmp_path)) == 6
    assert len(data.discover(tmp_path, ['marker_captures'])) == 3
    assert len(data.discover(marker, ['marker_captures'])) == 3
    assert len(data.discover(tmp_path, ['marker_captures', 'tree_captures'])) == 6
    assert data.discover(tmp_path, ['marker_capture']) == []  # exact name


def test_startup_remembers_and_validates(tmp_path):
    app = qt_app()
    preferences = tmp_path/'choices.json'
    dialog = StartDialog(preferences_path=preferences)
    dialog.open_workspace()
    assert dialog.result() != QDialog.Accepted and dialog.error.text()
    assert not preferences.exists()
    dialog.root.setText(str(tmp_path))
    dialog.filters.setText('marker_captures, tree_captures')
    dialog.reviewer.setText('reviewer2')
    dialog.session.setText(str(tmp_path/'annotations.json'))
    dialog.open_workspace()
    assert dialog.result() == QDialog.Accepted
    again = StartDialog(preferences_path=preferences)
    assert again.root.text() == str(tmp_path)
    assert again.filters.text() == 'marker_captures, tree_captures'
    assert again.reviewer.text() == 'reviewer2'
    assert again.session.text() == str(tmp_path/'annotations.json')
    assert again.profile.currentData() == 'pears'
    again.close()


def test_scope_restore_preserves_other_records(tmp_path):
    app = qt_app()
    marker = tmp_path/'capture'/'marker_captures'
    other = tmp_path/'capture'/'tree_captures'
    generate(marker); generate(other)
    session = tmp_path/'work.json'
    window = Window(str(tmp_path), str(marker/'geometry.json'), str(session))
    window.click(20,30)
    window.frames.setCurrentRow(4)
    window.click(45,55)
    old_id = window.h.value['id']
    window.close()
    filtered = Window(str(tmp_path), session=str(session), folder_filters=['marker_captures'])
    assert len(filtered.paths) == 3
    assert any(r['id'] == old_id for r in filtered.session['records'])
    filtered.frames.setCurrentRow(2)
    last = str(filtered.paths[2])
    filtered.close()
    restored = Window(session=str(session))
    assert len(restored.paths) == 3
    assert restored.session['last_frame'] == last
    assert restored.frames.currentRow() == 2
    saved = state.load(session)['records']
    assert len(saved) == 3
    assert next(r for r in saved if r['id'] == old_id)['observations']['P0']['xy'] == [45,55]
    restored.close()


def test_visible_clicks_only_delete_and_restore(tmp_path):
    app = qt_app()
    from PySide6.QtWidgets import QComboBox, QGraphicsSimpleTextItem
    from marker_annotation import state
    generate(tmp_path/'capture')
    session = tmp_path/'session.json'
    w = Window(str(tmp_path/'capture'),str(tmp_path/'capture/geometry.json'),str(session))
    assert not any(box.findText('occluded') >= 0 for box in w.findChildren(QComboBox))
    w.click(123,234)
    assert w.h.value['observations']['P0'] == dict(xy=[123,234],visibility='visible',modality='intensity')
    assert 'clicked' in w.corners.item(0).text()
    w.h.value['status'] = 'accepted'
    w.delete()
    assert 'P0' not in w.h.value['observations']
    assert w.h.value['status'] == 'annotated'
    assert 'unmarked' in w.corners.item(0).text()
    w.undo()
    assert w.h.value['observations']['P0']['xy'] == [123,234]
    assert w.h.value['status'] != 'accepted'
    legacy = dict(xy=[20,30],visibility='occluded',modality='intensity')
    w.h.value['observations']['P1'] = legacy
    w.close()
    restored = Window(session=str(session))
    assert restored.h.value['observations']['P1'] == legacy
    assert 'unmarked' in restored.corners.item(1).text()
    assert 'P1' not in [item.text() for item in restored.viewer.scene().items() if isinstance(item,QGraphicsSimpleTextItem)]
    restored.corners.setCurrentRow(1)
    restored.click(40,50)
    assert state.click_status(restored.h.value['observations']['P1']) == 'clicked'
    restored.close()
