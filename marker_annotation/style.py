"""Restrained, consistent desktop appearance independent of the host theme."""
from PySide6.QtGui import QColor, QPalette, QFont


def apply_style(app):
    app.setStyle('Fusion')
    palette = QPalette()
    for role, color in [(QPalette.Window, '#f2f4f5'), (QPalette.WindowText, '#26343d'),
                        (QPalette.Base, '#ffffff'), (QPalette.AlternateBase, '#f5f7f8'),
                        (QPalette.Text, '#26343d'), (QPalette.Button, '#f8f9fa'),
                        (QPalette.ButtonText, '#26343d'), (QPalette.Highlight, '#267f98'),
                        (QPalette.HighlightedText, '#ffffff'), (QPalette.ToolTipBase, '#ffffff'),
                        (QPalette.ToolTipText, '#26343d')]:
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor('#8b969e'))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#8b969e'))
    app.setPalette(palette)
    app.setFont(QFont('DejaVu Sans', 10))
    app.setStyleSheet('''
        QWidget { font-size: 12px; }
        QMainWindow, QDialog { background: #f2f4f5; }
        QLabel#screenTitle { font-size: 20px; font-weight: 600; color: #223640; }
        QLabel#sectionTitle { font-weight: 600; color: #40525f; padding: 3px 0; }
        QLabel#muted { color: #687983; }
        QLabel#error { color: #a14a49; }
        QLineEdit, QComboBox, QPlainTextEdit, QListWidget {
            background: #ffffff; border: 1px solid #d4dce0; border-radius: 3px;
            selection-background-color: #267f98; selection-color: white;
        }
        QLineEdit, QComboBox { padding: 6px; min-height: 18px; }
        QLineEdit:focus, QComboBox:focus { border-color: #267f98; }
        QListWidget { padding: 3px; outline: none; }
        QListWidget::item { padding: 5px 7px; }
        QListWidget::item:hover { background: #edf3f5; }
        QListWidget::item:selected { background: #dcecf1; color: #174c5c; }
        QPushButton { background: #ffffff; color: #344953; border: 1px solid #ccd6dc;
                      border-radius: 3px; padding: 7px 12px; }
        QPushButton:hover { background: #eaf1f4; border-color: #9eafb8; }
        QPushButton:pressed { background: #dce8ed; }
        QPushButton#primary { background: #267f98; border-color: #267f98; color: white; }
        QPushButton#primary:hover { background: #206f86; }
        QPushButton:disabled { color: #95a1a9; background: #edf0f2; }
        QSplitter::handle { background: #e0e6e9; width: 1px; }
        QStatusBar { color: #687983; border-top: 1px solid #d4dce0; }
        QGraphicsView { border: 1px solid #cbd4da; border-radius: 2px; }
        QPlainTextEdit { padding: 7px; }
        QToolTip { color: #26343d; background: #ffffff; border: 1px solid #cbd4da; }
    ''')
