"""Records and atomic persistence; no Qt dependencies."""
import copy
import csv
import json
import os
import tempfile
import uuid
from pathlib import Path


def click_status(observation):
    return 'clicked' if observation.get('visibility') == 'visible' and observation.get('xy') is not None else 'unmarked'


def record(frame, reviewer='', repeat=False):
    return dict(id=str(uuid.uuid4()), frame=frame, reviewer=reviewer, repeat=repeat,
                note='', status='not started', observations={}, pose=None)


class History:
    def __init__(self, value):
        self.value = value
        self.past, self.future = [], []

    def edit(self, key, observation):
        self.past.append(copy.deepcopy(self.value))
        self.future.clear()
        if observation is None:
            self.value['observations'].pop(key, None)
        else:
            self.value['observations'][key] = observation
        self.value['pose'] = None
        self.value['status'] = 'annotated'

    def undo(self):
        if self.past:
            self.future.append(copy.deepcopy(self.value))
            self.value = self.past.pop()
            # A restored fit still requires a fresh human review.
            if self.value['status'] == 'accepted':
                self.value['status'] = 'annotated'

    def redo(self):
        if self.future:
            self.past.append(copy.deepcopy(self.value))
            self.value = self.future.pop()
            if self.value['status'] == 'accepted':
                self.value['status'] = 'annotated'


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load(path):
    return json.loads(Path(path).read_text())


def export_csv(path, session):
    with open(path, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['record_id', 'frame', 'repeat', 'reviewer', 'status', 'rmse_px', 'note'])
        for r in session['records']:
            writer.writerow([r['id'], r['frame'], r['repeat'], r['reviewer'], r['status'],
                             (r['pose'] or {}).get('rmse_px', ''), r['note']])
