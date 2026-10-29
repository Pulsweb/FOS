"""Airgapped FOS Deployment Trigger.

Instead of performing the full FOS deployment locally against the Fabric
platform (as `deploy_fos_old.py` does), this script prepares an offline /
airgapped deployment and hands off execution to the `Deploy_FOS` notebook
running inside the target workspace:

1. Ensure the target workspace and a temporary staging Lakehouse exist.
2. Compose enabled modules and upload flat `src/`, `config/` and `data/` archives into
   `<temp_lakehouse>/Files/deployment_files/`.
3. Upload `scripts/Deploy_FOS.ipynb` as a notebook item into the workspace.
4. Trigger a job run of that notebook with the parameter
   `local_deployment_lakehouse` set to the temp lakehouse name.

The `Deploy_FOS` notebook then reads the deployment files from the
Lakehouse (bypassing the GitHub download that is used for the standard
end-customer flow). This enables deployments from an internal repository
mirror that is not reachable from the Fabric runtime.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import yaml

from compose_release import compose_release

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CICD_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

DEPLOY_NOTEBOOK_FILE = SCRIPTS_DIR / "Deploy_FOS.ipynb"
DEPLOY_NOTEBOOK_ITEM = "Deploy_FOS"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

repo_config_path = CICD_DIR / "repo_config.yaml"
if not repo_config_path.exists():
    raise FileNotFoundError(f"repo_config.yaml not found at {repo_config_path}")

with open(repo_config_path, "r", encoding="utf-8") as f:
    repo_cfg = yaml.safe_load(f) or {}

trg_workspace_name = repo_cfg.get("workspace")
if not trg_workspace_name:
    raise ValueError("Workspace not defined in repo_config.yaml (key: workspace)")

capacity_name = repo_cfg.get("capacity")
# Name of the temporary Lakehouse used to stage the deployment files.
# Override via repo_config.yaml (key: deploy_lakehouse). Default matches the
# lakehouse used in existing airgapped deployments.
temp_lakehouse_name = repo_cfg.get("deploy_lakehouse", "LH_Temp")
selected_modules = (
    os.environ.get("FOS_MODULE_IDS")
    or os.environ.get("FOS_MODULE_ID")
    or None
)

# ---------------------------------------------------------------------------
# fab CLI helpers
# ---------------------------------------------------------------------------


def run_fab_command(command, capture_output=False, silently_continue=False, timeout=600):
    """Invoke the Fabric CLI with a single command string."""
    # Always capture stderr so we can inspect it and surface useful error messages.
    result = subprocess.run(
        ["fab", "-c", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    # Echo stdout to our own stream, but only when the caller does NOT want to
    # consume the value itself - otherwise the id / value would be printed twice.
    if result.stdout and not capture_output:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if not silently_continue and result.returncode != 0:
        raise Exception(
            f"fab command failed. exit_code='{result.returncode}'; "
            f"stderr='{(result.stderr or '').strip()}'; command='{command}'"
        )
    return result.stdout.strip() if capture_output else None


def _run_fab_verbose(command, timeout=600):
    """Run a fab command and return (returncode, stdout, stderr) so the caller
    can inspect transient error hints such as ``retry-after``.

    Streams both streams to the CI log so nothing is hidden.
    """
    result = subprocess.run(
        ["fab", "-c", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode, result.stdout or "", result.stderr or ""


def _looks_rate_limited(text):
    """Detect Fabric transient errors that ask us to back off."""
    t = text.lower()
    return (
        "retry-after" in t
        or "toomanyrequests" in t
        or "429" in t
        or "rate limit" in t
        or "unexpectederror" in t
    )


def _get_lakehouse_id():
    """Return the staging Lakehouse id if it exists and is provisioned, else None."""
    out = run_fab_command(
        f"get /{trg_workspace_name}.Workspace/{temp_lakehouse_name}.Lakehouse -q id",
        capture_output=True,
        silently_continue=True,
    )
    if not out:
        return None
    # Error messages from fab start with a bracketed error code like '[NotFound]'
    if out.startswith("[") or " " in out:
        return None
    return out


def _wait_for_lakehouse(target_exists, timeout=60):
    """Poll until the Lakehouse is (or is not) present. Short interval, fail-fast."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        exists = _get_lakehouse_id() is not None
        if exists == target_exists:
            return _get_lakehouse_id()
        time.sleep(4)
    return _get_lakehouse_id() if target_exists else None


# ---------------------------------------------------------------------------
# Deployment steps
# ---------------------------------------------------------------------------


def get_workspace_id():
    """Return the id of the target workspace.

    The workspace itself is expected to be created by an earlier step in the
    GitHub Action (or manually) - this script only stages files into it and
    triggers the deployment notebook.
    """
    print(f"[1/5] Resolving workspace '{trg_workspace_name}'")
    ws_id = run_fab_command(
        f"get /{trg_workspace_name}.Workspace -q id",
        capture_output=True,
        silently_continue=True,
    )
    if not ws_id or ws_id.startswith("["):
        raise Exception(
            f"Workspace '{trg_workspace_name}' not found. "
            f"It must be created before running this script."
        )
    print(f"        Workspace ID: {ws_id}")
    return ws_id


def ensure_temp_lakehouse():
    """Create a fresh staging Lakehouse and wait until it is provisioned.

    Handles two transient conditions:
      * A stale Lakehouse from a previous failed run (deleted first for a clean slate).
      * Fabric returning ``retry-after`` on create when the workspace / capacity
        is still ramping up. When we see this, we back off longer between
        attempts instead of hammering the API.
    """
    print(f"[2/5] Creating staging Lakehouse '{temp_lakehouse_name}'")

    # Best-effort delete of any leftover Lakehouse from a previous failed run.
    if _get_lakehouse_id() is not None:
        print("        Existing Lakehouse found, deleting for a clean slate")
        run_fab_command(
            f"rm /{trg_workspace_name}.Workspace/{temp_lakehouse_name}.Lakehouse -f",
            silently_continue=True,
        )
        _wait_for_lakehouse(target_exists=False, timeout=60)

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        print(f"        Attempt {attempt}/{max_attempts}: creating Lakehouse")
        _, out, err = _run_fab_verbose(
            f"create /{trg_workspace_name}.Workspace/{temp_lakehouse_name}.Lakehouse"
        )
        rate_limited = _looks_rate_limited(out + err)

        # Give Fabric a short moment, then check whether the item appeared
        time.sleep(10)
        lh_id = _wait_for_lakehouse(target_exists=True, timeout=30)
        if lh_id:
            print(f"        Lakehouse ready. ID: {lh_id}")
            return lh_id

        if attempt == max_attempts:
            break

        # If Fabric asked us to retry later, back off substantially longer
        wait = (60 if rate_limited else 20) * attempt
        reason = "Fabric rate-limited (retry-after)" if rate_limited else "item not visible yet"
        print(f"        {reason} - waiting {wait}s before next attempt")
        time.sleep(wait)

    raise Exception(
        f"Could not create Lakehouse '{temp_lakehouse_name}' after {max_attempts} attempts. "
        f"Fabric API kept returning transient errors - check capacity health and quotas."
    )


def _make_zip(source_dir: Path, zip_path: Path, arc_prefix: str):
    """Zip every file in ``source_dir`` into ``zip_path`` under ``arc_prefix/``."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in source_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(source_dir).as_posix()
                zf.write(p, f"{arc_prefix}/{rel}")


def upload_deployment_files():
    """Compose modules, then package three zips and upload them to the staging Lakehouse.

    We upload zips instead of individual files because the Fabric CLI cannot
    reliably handle intermediate path segments that end with a Fabric item
    suffix (``.Notebook``, ``.Report``, ``.SemanticModel``, ``.DataPipeline``).
    In addition to being correct, this is dramatically faster - 3 uploads
    instead of ~1200.
    """
    print("[3/5] Packaging and uploading deployment files")

    remote_base = (
        f"/{trg_workspace_name}.Workspace/{temp_lakehouse_name}.Lakehouse/"
        f"Files/deployment_files"
    )
    run_fab_command(f'mkdir "{remote_base}"', silently_continue=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        payload_dir = tmp_dir / "payload"
        summary = compose_release(
            REPO_ROOT,
            payload_dir,
            selected_modules=selected_modules,
        )
        print(
            f"        Composed modules: {', '.join(summary['modules'])} "
            f"({summary['items']} deployment entries)"
        )
        archives = [
            ("src.zip", payload_dir / "src", "src"),
            ("config.zip", payload_dir / "config", "config"),
            ("data.zip", payload_dir / "data", "data"),
        ]
        for name, src_folder, arc_prefix in archives:
            local_zip = tmp_dir / name
            print(f"        Packaging {src_folder.name}/ -> {name}")
            _make_zip(src_folder, local_zip, arc_prefix)
            size_mb = local_zip.stat().st_size / (1024 * 1024)
            print(f"          {name}: {size_mb:.1f} MB")
            print(f"        Uploading {name}")
            run_fab_command(
                f'cp "{local_zip.as_posix()}" "{remote_base}/{name}" -f'
            )


def upload_deploy_notebook():
    """Import the Deploy_FOS.ipynb file as a notebook item in the workspace.

    Fabric CLI's import command reliably accepts a ``.Notebook`` folder
    (containing ``.platform`` + ``notebook-content.ipynb``) but rejects a raw
    ``.ipynb`` file with ``[InvalidInput]``. We therefore build the folder
    structure on the fly in a temp directory and hand that to ``fab import``.
    """
    print(f"[4/5] Uploading notebook '{DEPLOY_NOTEBOOK_ITEM}' from {DEPLOY_NOTEBOOK_FILE.name}")
    if not DEPLOY_NOTEBOOK_FILE.exists():
        raise FileNotFoundError(f"Notebook not found: {DEPLOY_NOTEBOOK_FILE}")

    platform_json = {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {
            "type": "Notebook",
            "displayName": DEPLOY_NOTEBOOK_ITEM,
            "description": "FOS airgapped deployment orchestrator",
        },
        "config": {
            "version": "2.0",
            "logicalId": "00000000-0000-0000-0000-000000000000",
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        item_dir = Path(tmp) / f"{DEPLOY_NOTEBOOK_ITEM}.Notebook"
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / ".platform").write_text(
            json.dumps(platform_json, indent=4), encoding="utf-8"
        )
        (item_dir / "notebook-content.ipynb").write_bytes(
            DEPLOY_NOTEBOOK_FILE.read_bytes()
        )

        run_fab_command(
            f"import /{trg_workspace_name}.Workspace/{DEPLOY_NOTEBOOK_ITEM}.Notebook "
            f'-i "{item_dir.as_posix()}" -f'
        )


def run_deploy_notebook():
    """Trigger the Deploy_FOS notebook with the airgapped parameter set."""
    current_branch = os.environ["FOS_CURRENT_BRANCH"]
    current_repository = os.environ["GITHUB_REPOSITORY"]
    print(f"[5/5] Running notebook with local_deployment_lakehouse='{temp_lakehouse_name}', current_branch='{current_branch}', git_repository_full_name='{current_repository}'")
    params = json.dumps(
        {
            "parameters": {
                "local_deployment_lakehouse": {
                    "type": "String",
                    "value": temp_lakehouse_name
                },
                "current_branch": {
                    "type": "String",
                    "value": current_branch
                },
                "git_repository_full_name": {
                    "type": "String",
                    "value": current_repository,
                }
            }
        }
    )
    run_fab_command(
        f"job run {trg_workspace_name}.Workspace/{DEPLOY_NOTEBOOK_ITEM}.Notebook -i {params}",
        timeout=7200,
    )


def cleanup_temp_lakehouse():
    """Remove the staging Lakehouse. Best-effort - failures are logged only.

    Set the environment variable ``FOS_KEEP_LAKEHOUSE=1`` to skip the cleanup
    and keep the staging Lakehouse around for debugging.
    """
    if os.environ.get("FOS_KEEP_LAKEHOUSE", "").lower() in ("1", "true", "yes"):
        print(
            f"[cleanup] FOS_KEEP_LAKEHOUSE is set - keeping staging Lakehouse "
            f"'{temp_lakehouse_name}' for debugging"
        )
        return
    print(f"[cleanup] Removing staging Lakehouse '{temp_lakehouse_name}'")
    try:
        run_fab_command(
            f"rm /{trg_workspace_name}.Workspace/{temp_lakehouse_name}.Lakehouse -f",
            silently_continue=True,
        )
    except Exception as ex:
        # Cleanup must never mask the deployment result
        print(f"        WARNING: could not remove staging lakehouse: {ex}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    get_workspace_id()
    ensure_temp_lakehouse()
    success = False
    try:
        upload_deployment_files()
        upload_deploy_notebook()
        run_deploy_notebook()
        success = True
        print("\nAirgapped deployment finished successfully.")
    finally:
        # Only clean up on success - on failure keep the staging Lakehouse
        # (with uploaded files + notebook) around so it can be inspected /
        # rerun manually. It will be replaced on the next script invocation.
        if success:
            cleanup_temp_lakehouse()
        else:
            print(
                f"\n[cleanup] Skipping Lakehouse deletion because deployment did "
                f"not complete successfully - staging Lakehouse "
                f"'{temp_lakehouse_name}' kept for debugging. It will be "
                f"replaced on the next run."
            )
