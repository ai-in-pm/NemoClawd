from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


def emit(payload: dict, exit_code: int = 0) -> None:
    print(json.dumps(payload), flush=True)
    raise SystemExit(exit_code)


def resolve_nat_command() -> list[str] | None:
    nat_executable = shutil.which("nat")
    if nat_executable is not None:
        return [nat_executable]

    scripts_dir = Path(sys.executable).resolve().parent
    nat_name = "nat.exe" if os.name == "nt" else "nat"
    nat_candidate = scripts_dir / nat_name
    if nat_candidate.exists():
        return [str(nat_candidate)]

    if importlib.util.find_spec("nat") is not None:
        return [sys.executable, "-m", "nat"]

    return None


def health() -> None:
    nat_command = resolve_nat_command()
    emit(
        {
            "ok": True,
            "natAvailable": nat_command is not None,
            "natExecutable": None if nat_command is None else nat_command[0],
            "python": sys.executable,
            "pythonVersion": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        }
    )


def run_workflow(args: argparse.Namespace) -> None:
    nat_command = resolve_nat_command()
    if nat_command is None:
        emit(
            {
                "ok": False,
                "code": 127,
                "command": ["nat"],
                "cwd": args.nat_workdir,
                "stdout": "",
                "stderr": "The nat executable is not available in the active Python environment.",
            },
            127,
        )

    command = [*nat_command, "run", "--config_file", args.config_file, "--input", args.input]
    for extra_arg in args.nat_arg:
        command.append(extra_arg)

    completed = subprocess.run(
        command,
        cwd=args.nat_workdir,
        capture_output=True,
        text=True,
        check=False,
    )

    emit(
        {
            "ok": completed.returncode == 0,
            "code": completed.returncode,
            "command": command,
            "cwd": args.nat_workdir,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        completed.returncode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemoclawd-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config-file", required=True)
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--nat-workdir", default=os.getcwd())
    run_parser.add_argument("--nat-arg", action="append", default=[])

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        health()
        return

    if args.command == "run":
        run_workflow(args)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
