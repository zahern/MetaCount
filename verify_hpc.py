"""Verify installed package version + that the fix landed."""
import paramiko

HOST = "aqua.qut.edu.au"
PORT = 22
USER = "ahernz"
PASSWORD = "SandySponge@1"
CONDA_ENV = "zigenv"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

cmd = (
    f"source /home/{USER}/miniconda3/etc/profile.d/conda.sh && "
    f"conda activate {CONDA_ENV} && "
    "pip show metacountregressor 2>/dev/null | grep -i version && "
    "python - <<'PY'\n"
    "import os, importlib.metadata as m\n"
    "print('VERSION', m.version('metacountregressor'))\n"
    "import metacountregressor as M\n"
    "p = os.path.dirname(M.__file__)\n"
    "mh = os.path.join(p, 'main_hpc.py')\n"
    "mp = os.path.join(p, 'main_hpc_lc_patch.py')\n"
    "t = open(mh).read()\n"
    "print('main_hpc shim?', 'from metacountregressor import main_hpc' in t)\n"
    "print('  has unpack_params', 'def unpack_params' in t)\n"
    "print('  has build_jax_data', 'def build_jax_data' in t)\n"
    "t2 = open(mp).read()\n"
    "print('patch variance_reg', 'variance_reg' in t2)\n"
    "print('patch variance_floor', 'variance_floor' in t2)\n"
    "print('patch _variance_penalty', '_variance_penalty' in t2)\n"
    "PY"
)

stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
out = stdout.read().decode(errors="replace").strip()
err = stderr.read().decode(errors="replace").strip()
with open(r"C:\Users\ahernz\AppData\Local\Temp\opencode\hpc_out.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n---STDERR---\n" + err)
ssh.close()
print("written to hpc_out.txt")
