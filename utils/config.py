from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    graphs: Path
    models: Path
    notebooks: Path
    results: Path
    scripts: Path
    utils: Path


def get_project_paths(start_dir: str | os.PathLike | None = None) -> ProjectPaths:
    root = Path(start_dir or Path.cwd()).resolve()
    # Walk up until we find markers of the repo (requirements or README)
    for _ in range(5):
        if (root / "requirements.txt").exists() or (root / "README.md").exists():
            break
        if root.parent == root:
            break
        root = root.parent

    data = root / "data"
    graphs = root / "graphs"
    models = root / "models"
    notebooks = root / "notebooks"
    results = root / "results"
    scripts = root / "scripts"
    utils = root / "utils"

    for p in (data, graphs, models, notebooks, results, scripts, utils):
        p.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root=root,
        data=data,
        graphs=graphs,
        models=models,
        notebooks=notebooks,
        results=results,
        scripts=scripts,
        utils=utils,
    )


def getenv_str(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    return val if val is not None else default


MAPI_KEY = getenv_str("MAPI_KEY")

