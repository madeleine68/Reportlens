"""Upload the API + UI to a Hugging Face Docker Space (Phase 5).

Assembles only what the Space needs in a temporary folder (no PDFs, no data/, no .env):
    src/reportlens/   the code
    requirements.txt  the locked runtime dependencies, without the ingest group, with CPU torch
    Dockerfile        deploy/Dockerfile
    README.md         deploy/space_README.md (Hugging Face reads the Space's settings from it)

Settings come from .env.deploy (git-ignored), falling back to .env for the LLM values. Values
are never printed.
Usage:
    uv run python scripts/deploy_space.py --dry-run     # assemble and list files, upload nothing
    uv run python scripts/deploy_space.py --secrets     # set the Space's secrets and variables
    uv run python scripts/deploy_space.py               # upload the code (the Space rebuilds)
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ENV = ROOT / ".env.deploy"
# Stored encrypted by Hugging Face, readable only by the running Space.
SECRETS = [
    "LLM_API_KEY",
    "DATABASE_URL",
    "NEO4J_PASSWORD",
    "SQL_READER_PASSWORD",
    "DEMO_PASSWORD",
    "TAVILY_API_KEY",
]
# Visible in the Space's settings.
VARIABLES = [
    "LLM_BASE_URL",
    "LLM_MODEL_MAIN",
    "LLM_MODEL_FAST",
    "NEO4J_URI",
    "NEO4J_USER",
    "RATE_LIMIT_PER_HOUR",
    "DAILY_QUESTION_CAP",
    "WEB_SEARCH",
    "SELF_RAG",
    "ROUTED_CORRECTIVE",
]
REQUIRED = ["HF_TOKEN", "HF_SPACE", "LLM_API_KEY", "DATABASE_URL", "NEO4J_URI", "DEMO_PASSWORD"]


def deploy_values() -> dict[str, str]:
    """.env.deploy, with .env filling in what it leaves out (e.g. the LLM settings)."""
    local = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    deploy = {k: v for k, v in dotenv_values(DEPLOY_ENV).items() if v}
    return {**{k: local[k] for k in local if k.startswith("LLM_")}, **deploy}


def runtime_requirements() -> str:
    """The locked runtime dependencies for Linux, with CPU-only torch and no CUDA packages."""
    exported = subprocess.run(
        [
            "uv",
            "export",
            "--no-default-groups",
            "--no-hashes",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--quiet",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = [line for line in exported.splitlines() if not re.match(r"(nvidia-|triton)", line)]
    return "\n".join(re.sub(r"^torch==([\d.]+)", r"torch==\1+cpu", line) for line in lines) + "\n"


def assemble(folder: Path) -> list[Path]:
    """Copy what the Space needs into `folder`; return the files, relative to it."""
    shutil.copytree(
        ROOT / "src" / "reportlens",
        folder / "src" / "reportlens",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (folder / "requirements.txt").write_text(runtime_requirements())
    shutil.copy(ROOT / "deploy" / "Dockerfile", folder / "Dockerfile")
    shutil.copy(ROOT / "deploy" / "space_README.md", folder / "README.md")
    return sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())


def set_secrets(api, space: str, values: dict[str, str]) -> None:  # noqa: ANN001
    """Set the Space's secrets and variables from the deploy values (names only are printed)."""
    for name in SECRETS:
        if name in values:
            api.add_space_secret(space, name, values[name])
            print(f"secret   {name}")
    for name in VARIABLES:
        if name in values:
            api.add_space_variable(space, name, values[name])
            print(f"variable {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload ReportLens to a Hugging Face Space.")
    parser.add_argument("--dry-run", action="store_true", help="assemble and list, upload nothing")
    parser.add_argument("--secrets", action="store_true", help="set secrets and variables only")
    args = parser.parse_args()

    values = deploy_values()
    with tempfile.TemporaryDirectory() as tmp:
        files = assemble(Path(tmp))
        if args.dry_run:
            top = [f for f in files if f.parent == Path(".")]
            print(f"{len(files)} files would be uploaded: {', '.join(map(str, top))}, and")
            print(f"  {len(files) - len(top)} files under src/reportlens/ (code, prompts, schema)")
            missing = [n for n in REQUIRED if n not in values]
            print(f"Missing from .env.deploy: {', '.join(missing) or 'nothing'}")
            return 0

        if missing := [n for n in REQUIRED if n not in values]:
            print(f"Set these in .env.deploy first: {', '.join(missing)}")
            return 1
        from huggingface_hub import HfApi

        api = HfApi(token=values["HF_TOKEN"])
        space = values["HF_SPACE"]
        api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
        if args.secrets:
            set_secrets(api, space, values)
            return 0
        api.upload_folder(
            folder_path=tmp,
            repo_id=space,
            repo_type="space",
            commit_message="Deploy ReportLens",
            delete_patterns=["src/**"],  # files removed locally disappear from the Space too
        )
    print(f"Uploaded. Build log: https://huggingface.co/spaces/{space}?logs=build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
