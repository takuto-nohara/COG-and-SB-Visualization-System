from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import platform
from pathlib import Path
from typing import Iterable

BOOTSTRAP_MARKER = ".cogsb-gui-bootstrap"


def _run(command: list[str], cwd: Path, *, check: bool = False) -> int:
    proc = subprocess.run(command, cwd=str(cwd))
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)} (exit {proc.returncode})")
    return proc.returncode


def _path_dirs() -> list[Path]:
    entries: list[Path] = []
    seen: set[str] = set()
    for item in os.environ.get("PATH", "").split(os.pathsep):
        if not item:
            continue
        try:
            resolved = str(Path(item).resolve())
        except Exception:
            resolved = str(Path(item))
        if resolved.lower() in seen:
            continue
        seen.add(resolved.lower())
        entries.append(Path(resolved))
    return entries


def _writable_dirs() -> list[Path]:
    dirs: list[Path] = []
    for p in _path_dirs():
        if p.is_dir():
            test_file = p / "__cogsb_gui_path_test.tmp"
            try:
                test_file.write_text("x", encoding="utf-8")
            except Exception:
                continue
            try:
                test_file.unlink(missing_ok=True)
            except Exception:
                pass
            dirs.append(p)
    return dirs


def _is_temporary_path(directory: Path) -> bool:
    normalized = str(directory).replace("\\", "/").lower()
    return "/.codex/tmp/" in normalized


def _dedup_directories(directories: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        try:
            resolved = str(directory.resolve())
        except Exception:
            resolved = str(directory)
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(directory)
    return out


def _candidate_shim_dirs() -> list[Path]:
    stable_candidates = [
        Path.home() / ".local" / "bin",
        Path(sysconfig.get_path("scripts", scheme="nt_user")),
    ]
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        stable_candidates.append(Path(localappdata) / "Microsoft" / "WindowsApps")

    deduped = _dedup_directories((*stable_candidates, *_writable_dirs()))

    stable: list[Path] = []
    temporary: list[Path] = []
    for directory in deduped:
        if not directory.is_dir():
            continue
        if _is_temporary_path(directory):
            temporary.append(directory)
        else:
            stable.append(directory)

    # まず PATH やユーザー環境で残りやすい安定場所を優先し、
    # 最後に Codex が作成する一時領域を許容する。
    return stable + temporary


def _entry_files() -> list[str]:
    return ["cogsb-gui", "cogsb-gui.cmd", "cogsb-gui.bat"]


def _has_command_in_dir(directory: Path) -> bool:
    for name in _entry_files():
        if (directory / name).exists():
            return True
    return False


def _available_command() -> bool:
    return shutil.which("cogsb-gui") is not None


def _build_installer(root: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "--user",
        "--no-input",
        "--disable-pip-version-check",
    ]
    print("[cogsb-gui-launcher] pip install -e . --user を実行します ...")
    return _run(command, cwd=root, check=False)


def _needs_install(root: Path) -> bool:
    project_file = root / "pyproject.toml"
    marker = root / BOOTSTRAP_MARKER
    if not marker.exists():
        return True
    try:
        return project_file.stat().st_mtime > marker.stat().st_mtime
    except Exception:
        return True


def _mark_bootstrap(root: Path) -> None:
    marker = root / BOOTSTRAP_MARKER
    try:
        marker.write_text("ok", encoding="utf-8")
    except Exception:
        pass


def _shim_path_for_command(target: Path, root: Path) -> str:
    launcher = root / "cogsb-gui-launch.py"
    return (
        '@echo off\r\n'
        'setlocal\r\n'
        f'"{sys.executable}" "{launcher}" %*\r\n'
    )


def _write_shim(directory: Path, root: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "cogsb-gui.cmd"
    content = _shim_path_for_command(directory, root)
    try:
        if shim.exists():
            existing = shim.read_text(encoding="utf-8")
            if existing == content:
                return True
        shim.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def _ensure_user_script_dir_on_path() -> None:
    try:
        user_script = Path(sysconfig.get_path("scripts", scheme="nt_user"))
    except Exception:
        return
    if not user_script.exists():
        return

    normalized_path = {str(p).lower() for p in _path_dirs()}
    if str(user_script.resolve()).lower() in normalized_path:
        return

    print()
    print("※ユーザー領域のスクリプトが次の場所にあります。PowerShell では PATH に未登録です。")
    print(f"  {user_script}")
    os.environ["PATH"] = f"{user_script};{os.environ.get('PATH', '')}"
    print("  本シェルに一時的に反映しました。")

    if _persist_user_path(user_script):
        print("  ユーザー環境変数 PATH にも反映しました。新しい PowerShell では自動的に `cogsb-gui` が使えるようになります。")
        return

    print("  以下を実行すると、次回から `cogsb-gui` がすぐ使えます。")
    print(f'  $env:PATH = "{user_script};$env:PATH"')
    print("  [Environment]::SetEnvironmentVariable(\"Path\", \"{0};\" + [Environment]::GetEnvironmentVariable(\"Path\",\"User\"), \"User\")".format(user_script))


def _persist_user_path(path: Path) -> bool:
    if platform.system() != "Windows":
        return False

    try:
        import winreg  # type: ignore[import-not-found]
    except Exception:
        return False

    new_entry = str(path)
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ | winreg.KEY_SET_VALUE,
        ) as key:
            try:
                current, _ = winreg.QueryValueEx(key, "Path")
            except Exception:
                current = ""

            current_entries = [item.strip() for item in str(current).split(";") if item.strip()]
            lowered = {item.lower() for item in current_entries}
            if new_entry.lower() in lowered:
                return True

            updated = ";".join([new_entry, *current_entries])
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, updated)
            return True
    except Exception:
        return False


def _install_shim(root: Path) -> bool:
    path_dirs = _path_dirs()
    path_set = {str(p).lower() for p in path_dirs}

    for directory in _candidate_shim_dirs():
        if _write_shim(directory, root):
            if str(directory.resolve()).lower() in path_set:
                print(f"[cogsb-gui-launcher] コマンドのショートカットを作成しました: {directory}\\cogsb-gui.cmd")
            else:
                print(
                    "[cogsb-gui-launcher] コマンドのショートカットを作成しましたが、"
                    f"この場所は PATH にありません: {directory}\\cogsb-gui.cmd"
                )
                _ensure_user_script_dir_on_path()
            return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parent

    if _needs_install(root):
        if _build_installer(root) != 0:
            print("依存関係のインストール（pip install -e . --user）に失敗しました。", file=sys.stderr)
            return 1
        _mark_bootstrap(root)

    if not _available_command():
        if not _install_shim(root):
            print("[cogsb-gui-launcher] PATH 上に作成可能なショートカット先が見つかりませんでした。")
            _ensure_user_script_dir_on_path()

    # 最後に GUI を起動
    return _run([sys.executable, "-m", "cogsb.gui"], cwd=root, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
