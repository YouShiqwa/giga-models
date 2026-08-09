import os
import sys
from pathlib import Path


project_dir = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[4]
local_python_paths = [str(repo_root), str(project_dir)]
for path in reversed(local_python_paths):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

# ``Launcher`` starts fresh Python workers, so sys.path alone is insufficient.
# Propagate the same precedence to FSDP workers through their environment.
existing_python_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
existing_python_paths = [path for path in existing_python_paths if path and path not in local_python_paths]
os.environ['PYTHONPATH'] = os.pathsep.join(local_python_paths + existing_python_paths)

import tyro  # noqa: E402
from giga_train import launch_from_config, setup_environment  # noqa: E402


def train(config: str):
    setup_environment()
    launch_from_config(config)


if __name__ == '__main__':
    tyro.cli(train)
