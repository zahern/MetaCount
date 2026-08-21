"""
Upload updated LC files to aquarius01 and submit all 4 PBS jobs.

What this does:
  1. Connects to aqua.qut.edu.au (aquarius01) via SSH
  2. Upgrades metacountregressor in the zigenv conda environment
  3. Uploads the latest run_lc_search_class_specific.py + all 4 PBS scripts
  4. Submits all 4 PBS jobs (2-, 3-, 4-, 5-class)
  5. Prints the queue status
"""
import paramiko
import os

HOST = "aqua.qut.edu.au"
PORT = 22
USER = "ahernz"
PASSWORD = "SandySponge@1"

REMOTE_DIR = "/mnt/hpccs01/home/ahernz/latent_class_metacount"
CONDA_ENV  = "zigenv"

LOCAL_BASE = r"Z:\latent_class_metacount"

FILES_TO_UPLOAD = [
    "run_lc_search_class_specific.py",
    "run_lc_2class.pbs",
    "run_lc_3class.pbs",
    "run_lc_4class.pbs",
    "run_lc_5class.pbs",
]

PBS_SCRIPTS = [
    "run_lc_2class.pbs",
    "run_lc_3class.pbs",
    "run_lc_4class.pbs",
    "run_lc_5class.pbs",
]


def run_cmd(ssh, cmd, timeout=120):
    """Run a remote command and return (stdout, stderr) as strings."""
    chan = ssh.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b""
    err = b""
    while True:
        if chan.recv_ready():
            out += chan.recv(4096)
        if chan.recv_stderr_ready():
            err += chan.recv_stderr(4096)
        if chan.exit_status_ready():
            # Drain any remaining output
            while chan.recv_ready():
                out += chan.recv(4096)
            while chan.recv_stderr_ready():
                err += chan.recv_stderr(4096)
            break
    return out.decode("utf-8", errors="replace").strip(), \
           err.decode("utf-8", errors="replace").strip()


print(f"Connecting to {HOST}:{PORT} as {USER} ...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
print("Connected.\n")

# ── 1. Install the NEWEST metacountregressor from GitHub (always latest) ────
#     We install straight from the git master HEAD so the HPC always gets the
#     code you just pushed -- PyPI only refreshes when a release is published,
#     and the publish workflow is gated on the (non-existent) `main` branch.
#     --force-reinstall guarantees a fresh copy even if the version number is
#     unchanged, and --no-deps avoids reinstalling the whole dependency tree.
print("Installing latest metacountregressor (git HEAD) into conda env '%s' ..." % CONDA_ENV)
GIT_URL = "https://github.com/zahern/MetaCount.git@master"
install_cmd = (
    f"source /home/{USER}/miniconda3/etc/profile.d/conda.sh && "
    f"conda activate {CONDA_ENV} && "
    f"pip install --upgrade --force-reinstall --no-deps "
    f"git+{GIT_URL} 2>&1"
)
out, err = run_cmd(ssh, install_cmd, timeout=600)
lines = [l for l in out.splitlines() if l.strip()]
for line in lines[-6:]:
    print(f"  {line}")
git_ok = ("Successfully installed" in out or "Successfully upgraded" in out) \
    and "ERROR" not in out and "fatal" not in out and "Could not find" not in out

if not git_ok:
    # Fallback: try PyPI (only useful once a release is published)
    print("  [warn] git install did not succeed; falling back to PyPI ...")
    fb = (
        f"source /home/{USER}/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"pip install --upgrade metacountregressor 2>&1"
    )
    out, err = run_cmd(ssh, fb, timeout=300)
    for line in [l for l in out.splitlines() if l.strip()][-6:]:
        print(f"  {line}")

if err:
    for line in err.splitlines()[-3:]:
        print(f"  [stderr] {line}")
print()

# ── 2. Ensure remote dir exists ──────────────────────────────────────────────
out, err = run_cmd(ssh, f"mkdir -p {REMOTE_DIR}")
if err:
    print(f"  mkdir stderr: {err}")

# ── 3. Upload files ───────────────────────────────────────────────────────────
sftp = ssh.open_sftp()
for fname in FILES_TO_UPLOAD:
    local_path = os.path.join(LOCAL_BASE, fname)
    remote_path = f"{REMOTE_DIR}/{fname}"
    print(f"  Uploading  {fname}  ->  {remote_path}")
    sftp.put(local_path, remote_path)
    # Strip Windows CR and set perms
    run_cmd(ssh, f"sed -i 's/\\r//' {remote_path}")
    run_cmd(ssh, f"chmod 644 {remote_path}")
sftp.close()
print("\nAll files uploaded.\n")

# ── 4. Submit PBS jobs ────────────────────────────────────────────────────────
print("Submitting PBS jobs ...")
for pbs in PBS_SCRIPTS:
    out, err = run_cmd(ssh, f"cd {REMOTE_DIR} && qsub {pbs}")
    job_id = out.strip()
    print(f"  qsub {pbs:25s}  ->  {job_id or '(no output)'}")
    if err:
        print(f"    stderr: {err}")

# ── 5. Show queue ─────────────────────────────────────────────────────────────
print("\nCurrent queue (ahernz):")
out, err = run_cmd(ssh, "qstat -u ahernz 2>/dev/null || qstat -u ahernz")
print(out if out else "  (no jobs in queue yet)")

ssh.close()
print("\nDone.")
