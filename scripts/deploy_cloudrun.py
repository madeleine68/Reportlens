"""Deploy the API + UI to Google Cloud Run (Phase 5).

The image is built by Cloud Build from the same small folder deploy_space.py assembles (code,
requirements.txt, Dockerfile), so nothing large is uploaded and no local Docker is needed.
The service: 2 GiB, 1 CPU, scales to zero when idle, at most one instance (the per-visitor
hourly limit is counted in memory), public. Secrets live in Secret Manager; everything else is
a plain environment variable.

Settings come from .env.deploy (git-ignored; GCP_PROJECT, optionally GCP_REGION), falling back
to .env for the LLM values. Values are never printed or put on a command line.
Usage, in order (needs `gcloud auth login` first):
    uv run python scripts/deploy_cloudrun.py --setup      # once: APIs, image registry, roles
    uv run python scripts/deploy_cloudrun.py --secrets    # store/refresh secrets
    uv run python scripts/deploy_cloudrun.py              # build and deploy; prints the URL
    uv run python scripts/deploy_cloudrun.py --dry-run    # show what would be deployed
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deploy_space import SECRETS, VARIABLES, assemble, deploy_values  # noqa: E402

SERVICE = "reportlens"
REPOSITORY = "reportlens"  # Artifact Registry repository for the image
DEFAULT_REGION = "us-east1"  # near Neon's and AuraDB's us-east
REQUIRED = ["GCP_PROJECT", "LLM_API_KEY", "DATABASE_URL", "NEO4J_URI", "DEMO_PASSWORD"]
APIS = [
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
]
# The compute service account runs both the builds and the service.
ROLES = ["roles/cloudbuild.builds.builder", "roles/secretmanager.secretAccessor"]


def gcloud(
    *args: str, check: bool = True, quiet: bool = False, **kwargs
) -> subprocess.CompletedProcess:
    """Run a gcloud command; with quiet=True its output is captured, not shown."""
    if quiet:
        kwargs.setdefault("capture_output", True)
    return subprocess.run(["gcloud", *args], check=check, text=True, **kwargs)


def setup(project: str, region: str) -> None:
    """Enable the APIs, create the image registry and let the service account read secrets."""
    gcloud("services", "enable", *APIS, "--project", project)
    exists = gcloud(
        "artifacts", "repositories", "describe", REPOSITORY,
        "--location", region, "--project", project, check=False, quiet=True,
    )  # fmt: skip
    if exists.returncode:
        gcloud(
            "artifacts", "repositories", "create", REPOSITORY, "--repository-format", "docker",
            "--location", region, "--project", project,
        )  # fmt: skip
    number = gcloud(
        "projects", "describe", project, "--format", "value(projectNumber)", quiet=True
    ).stdout.strip()
    account = f"serviceAccount:{number}-compute@developer.gserviceaccount.com"
    for role in ROLES:
        gcloud(
            "projects", "add-iam-policy-binding", project, "--member", account, "--role", role,
            "--condition", "None", quiet=True,
        )  # fmt: skip
        print(f"role     {role}")


def store_secrets(project: str, values: dict[str, str]) -> None:
    """Create each secret if missing and add the current value as its newest version (stdin)."""
    for name in SECRETS:
        if name not in values:
            continue
        found = gcloud("secrets", "describe", name, "--project", project, check=False, quiet=True)
        if found.returncode:
            gcloud(
                "secrets", "create", name, "--replication-policy", "automatic",
                "--project", project, quiet=True,
            )  # fmt: skip
        gcloud(
            "secrets", "versions", "add", name, "--data-file", "-", "--project", project,
            input=values[name], quiet=True,
        )  # fmt: skip
        print(f"secret   {name}")


def env_file(values: dict[str, str], folder: Path) -> Path:
    """The plain (non-secret) variables as a YAML file for --env-vars-file."""
    path = folder / "env.yaml"
    lines = [f"{n}: {json.dumps(values[n])}" for n in VARIABLES if n in values]  # JSON is YAML
    path.write_text("\n".join(lines) + "\n")
    return path


def deploy(project: str, region: str, values: dict[str, str]) -> None:
    """Build the image with Cloud Build, then deploy it to Cloud Run and print the URL."""
    image = f"{region}-docker.pkg.dev/{project}/{REPOSITORY}/app:latest"
    secrets = ",".join(f"{n}={n}:latest" for n in SECRETS if n in values)
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "build"
        assemble(build)
        gcloud("builds", "submit", str(build), "--tag", image, "--timeout", "1800",
               "--project", project)  # fmt: skip
        gcloud(
            "run", "deploy", SERVICE, "--image", image, "--region", region, "--project", project,
            "--port", "7860", "--memory", "2Gi", "--cpu", "1", "--cpu-boost",
            "--min-instances", "0", "--max-instances", "1", "--timeout", "300",
            "--allow-unauthenticated", "--env-vars-file", str(env_file(values, Path(tmp))),
            "--set-secrets", secrets,
        )  # fmt: skip
    url = gcloud(
        "run", "services", "describe", SERVICE, "--region", region, "--project", project,
        "--format", "value(status.url)", quiet=True,
    ).stdout.strip()  # fmt: skip
    print(f"Deployed: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy ReportLens to Google Cloud Run.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--setup", action="store_true", help="once: APIs, registry, roles")
    mode.add_argument("--secrets", action="store_true", help="store or refresh the secrets")
    mode.add_argument("--dry-run", action="store_true", help="show what would be deployed")
    args = parser.parse_args()

    values = deploy_values()
    missing = [n for n in REQUIRED if n not in values]
    project, region = values.get("GCP_PROJECT", ""), values.get("GCP_REGION", DEFAULT_REGION)
    if args.dry_run:
        print(f"Service {SERVICE} in {project or '<GCP_PROJECT>'} / {region}")
        print(f"Secrets:   {', '.join(n for n in SECRETS if n in values)}")
        print(f"Variables: {', '.join(n for n in VARIABLES if n in values)}")
        print(f"Missing from .env.deploy: {', '.join(missing) or 'nothing'}")
        return 0
    if missing:
        print(f"Set these in .env.deploy first: {', '.join(missing)}")
        return 1
    if args.setup:
        setup(project, region)
    elif args.secrets:
        store_secrets(project, values)
    else:
        deploy(project, region, values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
