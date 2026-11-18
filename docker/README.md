# Docker

Build and start the project environment from the repository root:

```bash
bash docker/build.sh
bash docker/run.sh
```

Before starting, edit the data volume in `run.sh` to match your host dataset
location. The repository is mounted at `/workspace`; the configured host data
directory is mounted at `/data`. The image includes reconstruction and annotation
dependencies. Rebuild and recreate the container to pick up dependency changes.

The runner forwards X11 for desktop applications and requires a Linux X11 or
XWayland session with `DISPLAY` set and host `xhost` installed. Run application
commands from `/workspace` inside the container. Save sessions and outputs under
`/workspace` or `/data` to retain them after removing the container.

## Qt binding compatibility

Annotation pins PySide6 to 6.11.2. The 6.12.0 wheels have a reference-count bug
on Python 3.10/3.11 that can abort with `none_dealloc: deallocating None` during
normal GUI interaction. If an older container has this version installed, rebuild
with `bash docker/build.sh`, exit the old container, and recreate it with
`bash docker/run.sh`. To repair the currently running container instead:

```bash
python -m pip install 'PySide6==6.11.2'
python -m marker_annotation
```

The image rebuild makes the fix persistent across container recreation.
