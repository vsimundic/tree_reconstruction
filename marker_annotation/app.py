"""Qt shell; state, numerical estimation and persistence live in separate modules."""
import argparse
import os
import copy
from pathlib import Path
import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QPushButton, QFileDialog, QComboBox, QLabel, QLineEdit, QPlainTextEdit,
    QSplitter, QMessageBox, QCheckBox, QDialog)
from . import data, state
from .pose import estimate, depth_check
from .viewer import Viewer
from .startup import StartDialog
from .style import apply_style
from .diagram import MarkerDiagram, height_caption


class Window(QMainWindow):
    def __init__(self, root=None, geometry=None, session=None, profile=None, folder_filters=None, reviewer=''):
        super().__init__()
        self.setWindowTitle('Marker reference annotation')
        self.resize(1440, 900)
        self.g = None
        self.folder_filters = list(folder_filters or [])
        self.default_reviewer = reviewer
        self._skip_save = False
        self.paths = []
        self.histories = {}
        self.h = None
        self.output = Path(session or 'marker_annotations.json').resolve()
        self.session = dict(version=1, capture_root='', geometry=None, records=[], conventions={})
        self.modality = QComboBox(); self.modality.addItems(['intensity', 'depth'])
        self.frames = QListWidget(); self.frames.setMinimumWidth(230)
        self.corners = QListWidget(); self.corners.setMinimumWidth(220)
        self.diagram = MarkerDiagram()
        self.viewer = Viewer()
        self.overlay = QCheckBox('Projected CAD'); self.overlay.setChecked(True)
        self.aligned = QCheckBox('Confirmed pixel alignment')
        self.scale = QLineEdit(); self.scale.setPlaceholderText('Confirmed axial depth m/unit (optional)')
        self.reviewer = QLineEdit(reviewer); self.reviewer.setPlaceholderText('Reviewer identifier')
        self.note = QLineEdit(); self.note.setPlaceholderText('Optional review note')
        self.report = QPlainTextEdit(); self.report.setReadOnly(True)
        self.report.setMaximumWidth(360)
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central); layout.setContentsMargins(14,12,14,8); layout.setSpacing(12)
        toolbar = QHBoxLayout(); layout.addLayout(toolbar)
        for label, callback in [('Workspace…', self.open_dataset), ('Load geometry', self.open_geometry),
                                ('Restore session', self.restore), ('Export JSON / CSV', self.export),
                                ('New independent repeat', self.repeat)]:
            b = QPushButton(label); b.clicked.connect(callback); toolbar.addWidget(b)
        split = QSplitter(); layout.addWidget(split, 1)
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0,0,8,0)
        frame_title = QLabel('Frames'); frame_title.setObjectName('sectionTitle'); ll.addWidget(frame_title)
        self.scope_label = QLabel('No images loaded'); self.scope_label.setObjectName('muted'); self.scope_label.setWordWrap(True)
        ll.addWidget(self.scope_label); ll.addWidget(self.frames, 1); split.addWidget(left)
        middle = QWidget(); ml = QVBoxLayout(middle); ml.setContentsMargins(0,0,0,0)
        row = QHBoxLayout(); row.addWidget(self.modality); row.addWidget(self.overlay)
        fit = QPushButton('Fit image'); fit.clicked.connect(self.viewer.fit); row.addWidget(fit)
        ml.addLayout(row); ml.addWidget(self.viewer, 1); split.addWidget(middle)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(8,0,0,0)
        corner_title = QLabel('Marker corners'); corner_title.setObjectName('sectionTitle')
        for widget in [corner_title, self.diagram, self.corners,
                       self.aligned, self.scale, self.reviewer, self.note, self.report]:
            rl.addWidget(widget)
        split.addWidget(right); split.setSizes([240, 800, 380])
        bottom = QHBoxLayout(); layout.addLayout(bottom)
        for label, callback in [('Previous', lambda: self.advance(-1)), ('Next', lambda: self.advance(1)),
                                ('Estimate pose', self.solve), ('Accept', self.accept),
                                ('Flag ambiguous', self.flag), ('Delete click', self.delete),
                                ('Undo', self.undo), ('Redo', self.redo)]:
            b = QPushButton(label); b.clicked.connect(callback)
            if label == 'Estimate pose': b.setObjectName('primary')
            bottom.addWidget(b)
        for shortcut, callback in [('Left', lambda: self.advance(-1)), ('Right', lambda: self.advance(1)),
                                   ('Ctrl+Z', self.undo), ('Ctrl+Shift+Z', self.redo), ('Delete', self.delete),
                                   ('E', self.solve), ('A', self.accept), ('F', self.viewer.fit)]:
            action = QAction(self); action.setShortcut(QKeySequence(shortcut)); action.triggered.connect(callback); self.addAction(action)
        self.frames.currentRowChanged.connect(self.select_frame)
        self.corners.currentRowChanged.connect(self.select_corner)
        self.viewer.clicked.connect(self.click)
        self.diagram.corner_selected.connect(self.diagram_click)
        self.diagram.orientation_changed.connect(self.save_diagram_view)
        self.modality.currentTextChanged.connect(self.redraw)
        self.overlay.toggled.connect(self.redraw)
        self.aligned.toggled.connect(self.conventions_changed)
        self.scale.editingFinished.connect(self.conventions_changed)
        self.reviewer.textEdited.connect(self.metadata); self.note.textEdited.connect(self.metadata)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.persist)
        self.statusBar().showMessage('Open a dataset and load a corner geometry JSON. Wheel: zoom · drag: pan · Shift-drag: relocate')
        if profile == 'pears':
            geometry = geometry or str(Path(__file__).with_name('helios_corners.json'))
            self.aligned.setText('Pixel alignment assumed · pears')
            self.aligned.setChecked(True)
            self.scale.setText('0.001')
            self.session['conventions'] = dict(aligned=True, depth_scale='0.001',
                alignment_basis='User authorized same-sensor pixel alignment assumption',
                depth_basis='Axial Z millimetres: paired pointcloud comparison on A/tree_0_V_0000 frames 0, 3, 7; max error 0.00075 m',
                invalid_depth='zero or nonfinite')
        if geometry and not (session and Path(session).exists()): self.load_geometry(geometry)
        if session and Path(session).exists(): self.restore_path(session)
        if root: self.load_dataset(root, folder_filters)
        if not root and not self.paths:
            self.viewer.scene().addText('Open a workspace to begin.').setDefaultTextColor(Qt.lightGray)

    def error(self, e):
        QMessageBox.warning(self, 'Action could not be completed', str(e))

    def open_dataset(self):
        self.persist()
        initial = dict(root=self.session['capture_root'], session=str(self.output),
                       folder_filters=self.folder_filters, reviewer=self.reviewer.text())
        dialog = StartDialog(self, initial=initial)
        if dialog.exec() == QDialog.Accepted:
            self.next_window = Window(**dialog.values)
            self.next_window.show()
            self._skip_save = True
            self.close()

    def load_dataset(self, root, folder_filters=None):
        self.h = None
        root = str(Path(root).resolve())
        if folder_filters is not None:
            self.folder_filters = list(folder_filters)
        self.session.update(capture_root=root, folder_filters=self.folder_filters)
        self.paths = data.discover(root, self.folder_filters)
        self.frames.clear()
        latest = {r['frame']: r for r in self.session['records']}
        for p in self.paths:
            status = latest.get(str(p), {}).get('status', 'not started')
            self.frames.addItem(str(p.relative_to(root)) + ' · ' + status)
        scope = ', '.join(self.folder_filters) or 'All folders'
        self.scope_label.setText(f'{len(self.paths)} frames · {scope}')
        self.scope_label.setToolTip(root)
        if self.paths:
            last = self.session.get('last_frame')
            index = next((i for i,p in enumerate(self.paths) if str(p) == last), 0)
            self.frames.setCurrentRow(index)
        else:
            self.viewer.scene().clear()
            self.viewer.scene().addText('No matching images. Change the folder or filter in Workspace…').setDefaultTextColor(Qt.lightGray)
            self.viewer.setSceneRect(self.viewer.scene().itemsBoundingRect()); self.viewer.fit()
            self.report.setPlainText('No *_intensity.png images match this workspace. Folder names must match exactly.')

    def open_geometry(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Corner geometry', filter='JSON (*.json)')
        if path: self.load_geometry(path)

    def load_geometry(self, path):
        try:
            self.g = data.geometry(path)
            for r in self.session['records']:
                r['pose'] = None
                if r['status'] == 'accepted': r['status'] = 'annotated'
            self.histories.clear()
            if self.h: self.h.value['pose'] = None
            self.session['geometry'] = dict(path=str(Path(path).resolve()), definition=self.g)
            self.corners.clear()
            self.keys = list(self.g['corners'])
            self.diagram.set_geometry(self.g)
            for key, c in self.g['corners'].items(): self.corners.addItem(f"{c['group']} / {key}")
            self.corners.setCurrentRow(0)
        except Exception as e: self.error(e)

    def select_frame(self, index):
        if index < 0: return
        try:
            self.h = None
            p = self.paths[index]
            self.intensity, self.depth, self.cal, calpath = data.read_frame(p)
            key = str(p.resolve())
            records = [r for r in self.session['records'] if r['frame'] == key]
            if not records:
                r = state.record(key, self.default_reviewer or self.reviewer.text()); r['calibration'] = dict(path=calpath, definition=self.cal)
                self.session['records'].append(r)
            else: r = records[-1]
            self.h = self.histories.setdefault(r['id'], state.History(r))
            self.session['last_frame'] = key
            self.note.setText(self.h.value['note']); self.reviewer.setText(self.h.value['reviewer'])
            self.redraw(); self.viewer.fit()
        except Exception as e: self.error(e)

    def selected(self):
        row = self.corners.currentRow()
        return self.keys[row] if self.g and row >= 0 else None

    def select_corner(self, *_):
        if self.g:
            self.diagram.set_selection(self.selected(), self.h.value['observations'] if self.h else {})
        self.redraw()

    def save_diagram_view(self, rotation, mirrored):
        self.session['diagram_view'] = dict(rotation=rotation, mirrored=mirrored)
        self.timer.start(350)

    def diagram_click(self, key):
        if self.g and key in self.keys:
            self.corners.setCurrentRow(self.keys.index(key))

    def click(self, x, y):
        if not self.h or not self.selected(): return
        self.h.edit(self.selected(), dict(xy=[x,y], visibility='visible', modality=self.modality.currentText()))
        self.changed()

    def delete(self):
        if self.h and self.selected() in self.h.value['observations']:
            self.h.edit(self.selected(), None)
            self.changed()

    def undo(self):
        if self.h: self.h.undo(); self.changed()

    def redo(self):
        if self.h: self.h.redo(); self.changed()

    def conventions_changed(self):
        for h in self.histories.values():
            h.value['pose'] = None
            if h.value['status'] == 'accepted': h.value['status'] = 'annotated'
        self.session['conventions'].update(aligned=self.aligned.isChecked(), depth_scale=self.scale.text())
        self.changed()

    def metadata(self):
        if self.h:
            self.h.value.update(reviewer=self.reviewer.text(), note=self.note.text()); self.timer.start(350)

    def changed(self):
        if self.h:
            self.session['records'] = [self.h.value if r['id']==self.h.value['id'] else r for r in self.session['records']]
            row = self.frames.currentRow()
            self.frames.item(row).setText(str(self.paths[row].relative_to(self.session['capture_root']))+' · '+self.h.value['status'])
        self.redraw(); self.timer.start(350)

    def redraw(self, *_):
        if not self.h: return
        try:
            mode = self.modality.currentText()
            rgb = data.display_image(self.depth if mode=='depth' else self.intensity, mode=='depth')
            obs = self.h.value['observations']
            if self.g:
                for index, key in enumerate(self.keys):
                    status = state.click_status(obs.get(key, {}))
                    c = self.g['corners'][key]
                    self.corners.item(index).setText(f"{c['group']} / {key} · Z {c['xyz'][2]*1000:g} mm · {status}")
                    self.corners.item(index).setToolTip(height_caption(self.g, c['xyz'][2]))
            shown = {k:o for k,o in obs.items() if state.click_status(o)=='clicked'
                     and (self.aligned.isChecked() or o.get('modality')==mode)}
            pose = self.h.value['pose']; projected = None
            if pose and self.g and self.overlay.isChecked() and (pose['modality']==mode or self.aligned.isChecked()):
                points = np.array([c['xyz'] for c in self.g['corners'].values()], float)
                uv = cv2.projectPoints(points, np.array(pose['rvec']), np.array(pose['tvec_m']), np.array(self.cal['k']).reshape(3,3), np.array(self.cal['d']))[0].reshape(-1,2)
                projected = dict(zip(self.keys, uv.tolist()))
            self.viewer.show_image(rgb, shown, self.selected(), projected, self.g.get('edges',[]) if self.g else [])
            lines = [f"Review: {self.h.value['status']}", 'Repeat: previous records hidden' if self.h.value['repeat'] else 'Primary annotation']
            if pose:
                lines += [f"RMSE: {pose['rmse_px']:.3f} px", f"Planar: {pose['planar']} · Ambiguous: {pose['ambiguous']}", 'Used: '+', '.join(pose['used']), 'Excluded: '+str(pose['excluded'])]
                lines += [f'{k}: {v:.3f} px' for k,v in pose['residuals_px'].items()]
                lines += [f'Depth {k}: {v}' for k,v in pose['depth'].items()]
            if not self.aligned.isChecked(): lines.append('Cross-image transfer disabled: alignment unconfirmed.')
            o = shown.get(self.selected(), {})
            if o.get('xy') and self.scale.text() and (mode=='depth' or self.aligned.isChecked()):
                lines.append(str(depth_check(self.depth, o['xy'], float(self.scale.text()))))
            self.report.setPlainText('\n'.join(lines))
            if self.g:
                self.diagram.set_selection(self.selected(), obs)
        except Exception as e: self.report.setPlainText(str(e))

    def solve(self):
        if not self.h or not self.g: self.error('Load images and a geometry JSON first.'); return
        try:
            scale = float(self.scale.text()) if self.scale.text() else None
            if scale is not None and scale <= 0: raise ValueError('Depth scale must be positive.')
            self.h.value['pose'] = estimate(self.g, self.h.value['observations'], self.cal, self.modality.currentText(), self.aligned.isChecked(), self.depth, scale)
            self.h.value['status'] = 'annotated'; self.changed()
        except Exception as e: self.error(e)

    def accept(self):
        if not self.h: return
        if not self.h.value['pose']: self.error('Estimate a pose before accepting.'); return
        if self.h.value['pose']['ambiguous']: self.error('Pose has competing solutions. Add nonplanar visible corners or flag ambiguous.'); return
        if not self.reviewer.text().strip(): self.error('Enter a reviewer identifier.'); return
        self.h.value['status'] = 'accepted'; self.changed(); self.persist(); self.advance(1)

    def flag(self):
        if self.h: self.h.value['status'] = 'ambiguous'; self.changed(); self.persist(); self.advance(1)

    def advance(self, step):
        self.persist()
        self.frames.setCurrentRow(max(0, min(len(self.paths)-1, self.frames.currentRow()+step)))

    def repeat(self):
        if not self.h: return
        r = state.record(self.h.value['frame'], self.reviewer.text(), True)
        r['calibration'] = copy.deepcopy(self.h.value['calibration'])
        self.session['records'].append(r); self.h = state.History(r); self.histories[r['id']] = self.h
        self.note.clear(); self.changed()

    def persist(self):
        try:
            if self.session['records']: state.save(self.output, self.session)
        except Exception as e: self.error(f'Autosave failed at {self.output}: {e}')

    def restore(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Restore session', filter='JSON (*.json)')
        if path: self.restore_path(path)

    def restore_path(self, path):
        try:
            self.h = None
            restored = state.load(path); self.session = restored; self.output = Path(path); self.histories.clear()
            if restored['geometry']:
                self.g = restored['geometry']['definition']
                mapping = data.normalize_helios_geometry(self.g)
                data.migrate_helios_records(restored['records'],mapping)
                if mapping:
                    restored.setdefault('corner_id_migrations',[]).append(dict(target=self.g['corner_id_schema'],mapping=mapping))
                self.keys=list(self.g['corners']); self.corners.clear()
                self.diagram.set_geometry(self.g)
                for k,c in self.g['corners'].items(): self.corners.addItem(c['group']+' / '+k)
                self.corners.setCurrentRow(0)
            view = restored.get('diagram_view', {})
            self.diagram.blockSignals(True)
            self.diagram.set_orientation(view.get('rotation',0),view.get('mirrored',False))
            self.diagram.blockSignals(False)
            conventions = restored.get('conventions', {})
            self.aligned.blockSignals(True); self.aligned.setChecked(conventions.get('aligned',False)); self.aligned.blockSignals(False)
            self.scale.setText(conventions.get('depth_scale',''))
            self.folder_filters = restored.get('folder_filters', [])
            self.load_dataset(restored['capture_root'])
            for i,p in enumerate(self.paths):
                records=[r for r in restored['records'] if r['frame']==str(p.resolve())]
                if records: self.frames.item(i).setText(str(p.relative_to(restored['capture_root']))+' · '+records[-1]['status'])
        except Exception as e: self.error(e)

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export complete annotations', 'annotations.json', 'JSON (*.json)')
        if path:
            try: state.save(path,self.session); state.export_csv(str(Path(path).with_suffix('.csv')),self.session)
            except Exception as e: self.error(e)

    def closeEvent(self, event):
        self.timer.stop()
        if not self._skip_save: self.persist()
        event.accept()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset'); parser.add_argument('--geometry'); parser.add_argument('--session')
    parser.add_argument('--profile', choices=['pears'], help='Real CAD, assumed alignment, verified millimetre axial depth')
    args=parser.parse_args()
    from PySide6 import __version__ as pyside_version
    from shiboken6 import __version__ as shiboken_version
    if sys.version_info < (3, 12) and '6.12.0' in (pyside_version, shiboken_version):
        parser.error(
            'PySide6/Shiboken 6.12.0 has a native reference-count bug on Python 3.10/3.11. '
            'Rebuild the Docker image and recreate the container, or run: '
            'python -m pip install "PySide6==6.11.2"'
        )
    # OpenCV's GUI wheel can export a Qt5 plugin directory during import.
    # PySide6 must load its own Qt6 platform plugins instead.
    from PySide6.QtCore import QLibraryInfo
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(Path(QLibraryInfo.path(QLibraryInfo.PluginsPath)) / 'platforms')
    if '/cv2/' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR', None)
    app=QApplication(sys.argv)
    apply_style(app)
    if not any((args.dataset, args.geometry, args.session, args.profile)):
        dialog = StartDialog()
        if dialog.exec() != QDialog.Accepted: return
        w = Window(**dialog.values)
    else:
        w = Window(args.dataset,args.geometry,args.session,args.profile)
    w.show(); sys.exit(app.exec())
