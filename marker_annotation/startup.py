"""Persistent startup choices, shared by first launch and workspace switching."""
from pathlib import Path
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QDialogButtonBox, QWidget)
from . import data, state

PREFERENCES = Path(__file__).resolve().parents[1] / 'annotation_sessions' / 'preferences.json'


def read_preferences(path=PREFERENCES):
    try:
        value = state.load(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class StartDialog(QDialog):
    def __init__(self, parent=None, preferences_path=PREFERENCES, initial=None):
        super().__init__(parent)
        self.preferences_path = Path(preferences_path)
        self.setWindowTitle('Open annotation workspace')
        self.setMinimumSize(740, 600)
        self.values = None
        saved = read_preferences(self.preferences_path)
        saved.update(initial or {})
        layout = QVBoxLayout(self); layout.setContentsMargins(28, 24, 28, 24); layout.setSpacing(18)
        title = QLabel('Marker reference annotation'); title.setObjectName('screenTitle'); layout.addWidget(title)
        subtitle = QLabel('Choose the images to review and where to save your work.')
        subtitle.setObjectName('muted'); layout.addWidget(subtitle)
        form = QFormLayout(); form.setSpacing(12); layout.addLayout(form)
        self.root = QLineEdit(saved.get('root', ''))
        self.root.setPlaceholderText('Example: /data/pears_original')
        self.filters = QLineEdit(', '.join(saved.get('folder_filters', [])))
        self.filters.setPlaceholderText('Example: A; marker_captures')
        self.session = QLineEdit(saved.get('session', str(PREFERENCES.parent / 'pears.json')))
        self.profile = QComboBox(); self.profile.addItem('Pears · Helios2 marker', 'pears'); self.profile.addItem('Custom geometry', 'custom')
        self.profile.setCurrentIndex(1 if saved.get('profile') == 'custom' else 0)
        self.geometry = QLineEdit(saved.get('geometry') or '')
        self.reviewer = QLineEdit(saved.get('reviewer', ''))
        self.add_path(form, 'Dataset folder', self.root, self.browse_root)
        form.addRow('Include folder names', self.filters)
        hint = QLabel('Example — only marker captures from A:\nDataset folder: /data/pears_original\nInclude folder names: A; marker_captures\n; means AND · , means OR (example: A,B; marker_captures).')
        hint.setObjectName('muted'); hint.setWordWrap(True); hint.setMinimumHeight(76); form.addRow('', hint)
        self.add_path(form, 'Session JSON', self.session, self.browse_session)
        form.addRow('Marker', self.profile)
        self.geometry_row = self.add_path(form, 'Corner geometry', self.geometry, self.browse_geometry)
        form.addRow('Reviewer', self.reviewer)
        self.profile.currentIndexChanged.connect(self.profile_changed); self.profile_changed()
        note = QLabel('An existing session resumes its saved annotations, marker geometry and conventions.\nThe selected dataset folder and filter control which frames are shown. Other records are retained.')
        note.setWordWrap(True); note.setMinimumHeight(48); note.setObjectName('muted'); layout.addWidget(note)
        self.error = QLabel(); self.error.setWordWrap(True); self.error.setObjectName('error'); layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.open_button = buttons.addButton('Open workspace', QDialogButtonBox.AcceptRole)
        self.open_button.setObjectName('primary'); self.open_button.setDefault(True)
        buttons.accepted.connect(self.open_workspace); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_path(self, form, label, field, browse):
        row = QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.addWidget(field, 1)
        button = QPushButton('Browse…'); button.clicked.connect(browse); row.addWidget(button)
        widget = QWidget()
        widget.setLayout(row); form.addRow(label, widget)
        return widget

    def browse_root(self):
        path = QFileDialog.getExistingDirectory(self, 'Select dataset folder', self.root.text())
        if path: self.root.setText(path)

    def browse_session(self):
        dialog = QFileDialog(self, 'Choose an existing or new session JSON', self.session.text(), 'JSON (*.json)')
        dialog.setAcceptMode(QFileDialog.AcceptSave); dialog.setOption(QFileDialog.DontConfirmOverwrite)
        dialog.setDefaultSuffix('json')
        if dialog.exec(): self.session.setText(dialog.selectedFiles()[0])

    def browse_geometry(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Corner geometry', self.geometry.text(), 'JSON (*.json)')
        if path: self.geometry.setText(path)

    def profile_changed(self, *_):
        self.geometry_row.setEnabled(self.profile.currentData() == 'custom')

    def open_workspace(self):
        try:
            root = Path(self.root.text().strip()).expanduser()
            if not self.root.text().strip() or not root.is_dir():
                raise ValueError('Choose an existing dataset folder.')
            expression = self.filters.text().strip()
            data.parse_folder_filters(expression)
            filters = [expression] if expression else []
            if not self.session.text().strip(): raise ValueError('Choose a session JSON path for autosave.')
            session = Path(self.session.text().strip()).expanduser().resolve()
            if session.exists():
                restored = state.load(session)
                if not isinstance(restored, dict) or 'records' not in restored or 'capture_root' not in restored:
                    raise ValueError('The selected JSON is not an annotation session.')
            custom = self.profile.currentData() == 'custom'
            geometry = self.geometry.text().strip()
            if custom and not session.exists():
                if not geometry: raise ValueError('Choose a corner geometry JSON for the custom marker.')
                data.geometry(Path(geometry).expanduser())
            self.values = dict(root=str(root.resolve()), folder_filters=filters,
                               session=str(session), profile='custom' if custom else 'pears',
                               geometry=str(Path(geometry).expanduser().resolve()) if custom and geometry else None,
                               reviewer=self.reviewer.text().strip())
            state.save(self.preferences_path, self.values)
            self.accept()
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.error.setText(str(e))
