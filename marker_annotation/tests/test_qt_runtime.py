"""Native binding regressions run in a child process so a crash is reported."""
import os
import subprocess
import sys
import textwrap


def test_repeated_qt_calls_and_mouse_moves_survive():
    script = textwrap.dedent('''
        import sys
        import numpy as np
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication, QLineEdit
        from marker_annotation.viewer import Viewer
        app = QApplication([])
        field = QLineEdit()
        viewer = Viewer()
        viewer.setMouseTracking(True)
        viewer.show_image(np.zeros((480, 640, 3), np.uint8), {}, None)
        viewer.show()
        app.processEvents()
        field.setText('warmup')
        before = sys.getrefcount(None)
        for i in range(1000):
            field.setText('reference-count check')
        delta = sys.getrefcount(None) - before
        assert abs(delta) < 20, f'Qt setters changed None refcount by {delta}'
        for i in range(10000):
            event = QMouseEvent(QMouseEvent.MouseMove, QPointF(40+i%200, 40),
                                QPointF(40+i%200, 40), Qt.NoButton,
                                Qt.NoButton, Qt.NoModifier)
            app.sendEvent(viewer.viewport(), event)
        viewer.close()
        app.processEvents()
        print('Qt interaction stress passed')
    ''')
    result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Qt interaction stress passed' in result.stdout


def test_affected_runtime_reports_repair(monkeypatch, capsys):
    import PySide6
    import shiboken6
    from marker_annotation.app import main
    import pytest
    monkeypatch.setattr(sys, 'argv', ['marker_annotation'])
    monkeypatch.setattr(sys, 'version_info', (3, 10, 22))
    monkeypatch.setattr(PySide6, '__version__', '6.12.0')
    monkeypatch.setattr(shiboken6, '__version__', '6.12.0')
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert 'python -m pip install "PySide6==6.11.2"' in capsys.readouterr().err
