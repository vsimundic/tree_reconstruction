from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QFont
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene


class Viewer(QGraphicsView):
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor('#202328'))
        self.start = None
        self.auto_fit = False
        self.marker_opacity = 0.45
        self.overlay_opacity = 0.35

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit: self.fit()

    def show_image(self, rgb, observations, selected, projected=None, edges=()):
        transform = self.transform()
        self.scene().clear()
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.scene().addPixmap(QPixmap.fromImage(image))
        self.setSceneRect(0, 0, w, h)
        projection_color = QColor(130,165,141,round(255*self.overlay_opacity))
        if projected:
            for a, b in edges:
                if a in projected and b in projected:
                    self.scene().addLine(*projected[a], *projected[b], QPen(projection_color, 0.8))
            for key, (x, y) in projected.items():
                self.scene().addEllipse(x-2, y-2, 4, 4, QPen(projection_color, 0.8))
        for key, o in observations.items():
            if o.get('xy') is None:
                continue
            x, y = o['xy']
            color = QColor('#59b7d2' if key == selected else '#dfbf80')
            color.setAlpha(round(255*self.marker_opacity))
            self.scene().addEllipse(x-2.5, y-2.5, 5, 5, QPen(color, 0.8))
            font = QFont(); font.setPixelSize(10)
            label = self.scene().addSimpleText(key, font)
            label.setBrush(color)
            label.setPos(x+5, y-14)
            if projected and key in projected:
                residual_color = QColor(color); residual_color.setAlpha(round(180*self.overlay_opacity))
                self.scene().addLine(x, y, *projected[key], QPen(residual_color, 0.6))
        self.setTransform(transform)

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        if 0.05 < self.transform().m11()*factor < 50:
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        self.start = event.position() if event.button() == Qt.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start is not None and (event.position()-self.start).manhattanLength() < 4:
            p = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(p):
                self.clicked.emit(p.x(), p.y())
        self.start = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        # Shift-drag precisely relocates the selected observation; plain drag pans.
        if event.buttons() & Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            p = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(p):
                self.clicked.emit(p.x(), p.y())
            return
        super().mouseMoveEvent(event)
