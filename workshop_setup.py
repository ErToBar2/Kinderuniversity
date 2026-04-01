from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable


REPO_MARKERS = ("1_Presentation", "2_Practical Exercise", "3_Game")


def _normalize_start(start: str | Path | None) -> Path:
    if start is None:
        path = Path.cwd().resolve()
    else:
        path = Path(start).resolve()
    return path.parent if path.is_file() else path


def find_repo_root(start: str | Path | None = None) -> Path:
    start_path = _normalize_start(start)
    for candidate in (start_path, *start_path.parents):
        if all((candidate / marker).exists() for marker in REPO_MARKERS):
            return candidate
    raise FileNotFoundError(
        "Could not find the Kinderuniversity repository root. "
        "Please open the notebook or script from inside the repository."
    )


def enter_repo_subdir(subdir: str, start: str | Path | None = None) -> tuple[Path, Path]:
    repo_root = find_repo_root(start)
    target_dir = repo_root / subdir
    if not target_dir.exists():
        raise FileNotFoundError(f"Could not find expected folder: {target_dir}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(target_dir)
    return repo_root, target_dir


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def check_required_packages(
    requirements: Iterable[dict[str, str] | tuple[str, str, str] | str],
    context_name: str,
    include_notebook_hint: bool = False,
) -> None:
    normalized: list[dict[str, str]] = []
    for requirement in requirements:
        if isinstance(requirement, str):
            normalized.append({"module": requirement, "pip": requirement, "purpose": ""})
        elif isinstance(requirement, tuple):
            module, pip_name, purpose = requirement
            normalized.append({"module": module, "pip": pip_name, "purpose": purpose})
        else:
            normalized.append(
                {
                    "module": requirement["module"],
                    "pip": requirement.get("pip", requirement["module"]),
                    "purpose": requirement.get("purpose", ""),
                    "install_hint": requirement.get("install_hint", ""),
                }
            )

    print(f"[setup] Checking Python packages for {context_name}...")
    missing = [item for item in normalized if importlib.util.find_spec(item["module"]) is None]
    if not missing:
        print(f"[setup] All required packages for {context_name} are available.")
        return

    print(f"[setup] Some packages for {context_name} are missing:")
    for item in missing:
        purpose_suffix = f" - {item['purpose']}" if item["purpose"] else ""
        display_name = item["pip"] or item["module"]
        print(f"  - {display_name} (import: {item['module']}){purpose_suffix}")
        if item.get("install_hint"):
            print(f"    {item['install_hint']}")

    install_packages = _dedupe(item["pip"] for item in missing if item.get("pip"))

    if install_packages:
        install_command = "python -m pip install " + " ".join(install_packages)
        print("[setup] Install them in your active environment with:")
        print("  python -m pip install --upgrade pip")
        print(f"  {install_command}")
    if include_notebook_hint:
        print("[setup] If Jupyter is not installed yet, also run:")
        print("  python -m pip install notebook")
    print("[setup] After installing, restart the Python process or Jupyter kernel and run the code again.")

    missing_names = ", ".join((item["pip"] or item["module"]) for item in missing)
    raise ModuleNotFoundError(f"Missing packages for {context_name}: {missing_names}")


def announce_first_run_models(models: Iterable[str], recommendation: str) -> None:
    model_list = list(models)
    if not model_list:
        return

    print("[setup] First-run note:")
    print("[setup] Ultralytics may download these model weights the first time you run this material:")
    for model_name in model_list:
        print(f"  - {model_name}")
    print(f"[setup] Recommendation: {recommendation}")


def resolve_model_path(model_dir: str | Path, model_name: str) -> str:
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    local_path = model_dir_path / model_name
    if local_path.exists():
        return str(local_path)

    print(
        f"[setup] {model_name} was not found in {model_dir_path}. "
        "Ultralytics will try to download it automatically on first use."
    )
    return model_name
