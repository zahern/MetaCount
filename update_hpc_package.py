"""
Refresh the metacountregressor install on the HPC (aquarius01) from PyPI.

Use this before submitting ANY job (QLD RP, latent-class, etc.) to make sure
the shared `zigenv` conda environment has your most recent push.  The publish
workflow bumps the version and uploads to PyPI on every push to master, so
`pip install --upgrade` here always grabs the newest commit.

Usage:
    python update_hpc_package.py
"""

import paramiko

HOST = "aqua.qut.edu.au"
PORT = 22
USER = "ahernz"
PASSWORD = "SandySponge@1"

CONDA_ENV = "zigenv"


def run_cmd(ssh, cmd, timeout=600):
    """Run a remote command and return (stdout, stderr) as strings."""
    chan = ssh.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out, err = b"", b""
    while True:
        if chan.recv_ready():
            out += chan.recv(4096)
        if chan.recv_stderr_ready():
            err += chan.recv_stderr(4096)
        if chan.exit_status_ready():
            while chan.recv_ready():
                out += chan.recv(4096)
            while chan.recv_stderr_ready():
                err += chan.recv_stderr(4096)
            break
    return out.decode("utf-8", errors="replace").strip(), \
           err.decode("utf-8", errors="replace").strip()


def main():
    print(f"Connecting to {HOST}:{PORT} as {USER} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("Connected.\n")

    print(f"Upgrading metacountregressor (PyPI) in '{CONDA_ENV}' ...")
    install_cmd = (
        f"source /home/{USER}/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"pip install --upgrade --index-url https://pypi.org/simple metacountregressor 2>&1"
    )
    out, err = run_cmd(ssh, install_cmd, timeout=300)
    for line in [l for l in out.splitlines() if l.strip()][-8:]:
        print(f"  {line}")

    if err:
        for line in err.splitlines()[-3:]:
            print(f"  [stderr] {line}")

    ssh.close()
    print("\nDone. HPC `zigenv` now has the latest metacountregressor from PyPI.")


if __name__ == "__main__":
    main()
