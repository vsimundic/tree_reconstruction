"""One CAD plan view with readable labels for all annotation corners."""
import math
from .state import click_status
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QCheckBox)


def height_caption(geometry, z):
    plate = [c['xyz'][2] for c in geometry['corners'].values() if c['group'] == 'plate']
    if plate and z > max(plate) + 1e-7:
        surface = 'Raised surface'
    elif plate and max(plate)-min(plate) > 1e-7 and abs(z-min(plate)) < 1e-7:
        surface = 'Plate underside'
    elif plate and abs(z-max(plate)) < 1e-7:
        surface = 'Plate top'
    else:
        surface = 'Surface'
    return f'{surface} · Z = {z*1000:g} mm'


class PlanView(QWidget):
    corner_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(330, 270)
        self.definition = None
        self.rotation = 0
        self.mirrored = False
        self.selected = None
        self.observations = {}
        self.points, self.labels = {}, {}
        self.setMouseTracking(True)

    def orient(self, x, y):
        # Screen coordinates: positive angles rotate clockwise; mirror is applied last.
        angle = math.radians(self.rotation)
        u = math.cos(angle)*x - math.sin(angle)*y
        v = math.sin(angle)*x + math.cos(angle)*y
        return (-u if self.mirrored else u), v

    def layout_corners(self):
        self.points, self.labels = {}, {}
        if not self.definition: return
        corners = self.definition['corners']
        xy = {key: self.orient(c['xyz'][0], -c['xyz'][1]) for key,c in corners.items()}
        xmin, xmax = min(p[0] for p in xy.values()), max(p[0] for p in xy.values())
        ymin, ymax = min(p[1] for p in xy.values()), max(p[1] for p in xy.values())
        margin = 78
        scale = min((self.width()-2*margin)/max(xmax-xmin, 1e-6),
                    (self.height()-68)/max(ymax-ymin, 1e-6))
        cx, cy = self.width()/2, self.height()/2
        for key,(x,y) in xy.items():
            self.points[key] = QPointF(cx+(x-(xmin+xmax)/2)*scale,
                                       cy+(y-(ymin+ymax)/2)*scale)
        active = list(corners)
        # Independent label rails preserve exact corner locations and keep IDs apart.
        ordered = sorted(active, key=lambda k: (self.points[k].x(), self.points[k].y()))
        sides = [ordered[:math.ceil(len(ordered)/2)], ordered[math.ceil(len(ordered)/2):]]
        for side, keys in enumerate(sides):
            keys.sort(key=lambda k: self.points[k].y())
            spacing = min(26, (self.height()-60)/max(len(keys),1))
            top = (self.height()-spacing*len(keys))/2
            for i,key in enumerate(keys):
                self.labels[key] = QRectF(8 if side==0 else self.width()-64,
                                          top+i*spacing, 56, spacing-2)

    def paintEvent(self, event):
        self.layout_corners()
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#f8fafb'))
        if not self.definition:
            p.setPen(QColor('#687983')); p.drawText(self.rect(), Qt.AlignCenter, 'Load a marker geometry to see its corners.'); return
        for a,b in self.definition.get('edges', []):
            if a not in self.points or b not in self.points: continue
            active = a in self.labels and b in self.labels
            p.setPen(QPen(QColor('#6d8591' if active else '#dce4e8'), 1.5 if active else 1))
            p.drawLine(self.points[a],self.points[b])
        font = QFont(self.font()); font.setPixelSize(12); font.setBold(True); p.setFont(font)
        for key, rect in self.labels.items():
            point = self.points[key]; selected = key == self.selected
            color = QColor('#167e9c' if selected else '#455c69')
            endpoint = QPointF(rect.right() if rect.center().x()<self.width()/2 else rect.left(),rect.center().y())
            p.setPen(QPen(QColor('#167e9c' if selected else '#adbcc4'), 1.5 if selected else 1))
            p.drawLine(point,endpoint)
            p.setBrush(QColor('#167e9c' if selected else '#ffffff')); p.setPen(QPen(color,2))
            p.drawEllipse(point,4.5 if selected else 3.5,4.5 if selected else 3.5)
            if selected:
                p.setPen(Qt.NoPen); p.setBrush(QColor('#dceef3')); p.drawRoundedRect(rect,3,3)
            p.setPen(color); p.drawText(rect,Qt.AlignCenter,key)
        p.setFont(self.font()); p.setPen(QColor('#687983'))
        caption = 'CAD plan · +X right · +Y up' if not self.rotation and not self.mirrored else f'CAD plan · {self.rotation}° clockwise'
        if self.mirrored: caption += ' · mirrored'
        p.drawText(QRectF(8,self.height()-24,self.width()-16,20),Qt.AlignCenter,caption)

    def mousePressEvent(self, event):
        self.layout_corners()
        pos = event.position()
        key = next((k for k,r in self.labels.items() if r.contains(pos)), None)
        if key is None and self.labels:
            nearest = min(self.labels,key=lambda k: (self.points[k]-pos).manhattanLength())
            if (self.points[nearest]-pos).manhattanLength() <= 14: key = nearest
        if key is not None: self.corner_selected.emit(key)

    def mouseMoveEvent(self, event):
        self.layout_corners()
        key = next((k for k,r in self.labels.items() if r.contains(event.position())),None)
        if key:
            c=self.definition['corners'][key]
            status=click_status(self.observations.get(key,{}))
            self.setToolTip(f"{key} · {c['group']}\n{height_caption(self.definition,c['xyz'][2])}\n{status}")
        else: self.setToolTip('')


class MarkerDiagram(QWidget):
    corner_selected = Signal(str)
    orientation_changed = Signal(int, bool)

    def __init__(self):
        super().__init__()
        self.definition = None
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(5)
        controls = QHBoxLayout(); controls.setSpacing(4)
        self.rotate_left = QPushButton('−90°'); self.rotate_right = QPushButton('+90°')
        self.rotation = QSpinBox(); self.rotation.setRange(0,359); self.rotation.setWrapping(True)
        self.rotation.setSingleStep(15); self.rotation.setSuffix('°')
        self.rotation.setToolTip('Diagram rotation clockwise. Type an angle or use the arrows.')
        self.mirror = QCheckBox('Mirror')
        self.mirror.setToolTip('Mirror the rotated diagram left to right. CAD coordinates and IDs stay unchanged.')
        self.reset = QPushButton('Reset')
        self.rotate_left.setToolTip('Rotate the diagram 90° counterclockwise')
        self.rotate_right.setToolTip('Rotate the diagram 90° clockwise')
        for widget in [self.rotate_left,self.rotate_right,self.rotation,self.mirror,self.reset]:
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.plan = PlanView(); layout.addWidget(self.plan)
        self.rotate_left.clicked.connect(lambda: self.rotation.setValue((self.rotation.value()-90)%360))
        self.rotate_right.clicked.connect(lambda: self.rotation.setValue((self.rotation.value()+90)%360))
        self.rotation.valueChanged.connect(self.update_orientation)
        self.mirror.toggled.connect(self.update_orientation)
        self.reset.clicked.connect(lambda: self.set_orientation(0,False))
        self.detail = QLabel('Select a corner in the diagram or list.'); self.detail.setWordWrap(True)
        self.detail.setMinimumHeight(34); layout.addWidget(self.detail)
        self.plan.corner_selected.connect(self.corner_selected)

    def set_orientation(self, rotation, mirrored):
        self.rotation.blockSignals(True); self.mirror.blockSignals(True)
        self.rotation.setValue(int(rotation)%360); self.mirror.setChecked(bool(mirrored))
        self.rotation.blockSignals(False); self.mirror.blockSignals(False)
        self.update_orientation()

    def update_orientation(self, *_):
        self.plan.rotation = self.rotation.value(); self.plan.mirrored = self.mirror.isChecked()
        self.plan.layout_corners(); self.plan.update()
        self.orientation_changed.emit(self.plan.rotation,self.plan.mirrored)

    def set_geometry(self, definition):
        self.definition = definition; self.plan.definition = definition
        self.plan.update()

    def set_selection(self, key, observations):
        if not self.definition or key not in self.definition['corners']: return
        self.plan.selected=key; self.plan.observations=observations
        c=self.definition['corners'][key]; z=c['xyz'][2]
        status=click_status(observations.get(key,{}))
        self.detail.setText(f"{key} · {c['group']} · {status}\n{height_caption(self.definition,z)}")
        self.plan.update()
