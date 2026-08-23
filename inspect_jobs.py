"""Compare valid HPC backups (md5) and show a valid .pbs wrapper template."""
import paramiko

HOST = "aqua.qut.edu.au"
PORT = 22
USER = "ahernz"
PASSWORD = "SandySponge@1"
REMOTE_DIR = "/mnt/hpccs01/home/ahernz/latent_class_metacount"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

cmd = (
    "echo '=== md5 of candidate 80480-byte scripts ===' && "
    "cd " + REMOTE_DIR + " && md5sum run_lc_patch.py _run_search_2class.py _run_search_3class.py _run_search_4class.py _run_search_5class.py run_lc_search_class_specific.py 2>&1; "
    "echo '=== head of run_lc_search.pbs (valid wrapper) ===' && cat " + REMOTE_DIR + "/run_lc_search.pbs 2>&1; "
    "echo '=== head of run_lc_search_enhanced.pbs ===' && cat " + REMOTE_DIR + "/run_lc_search_enhanced.pbs 2>&1"
)

stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
out = stdout.read().decode(errors="replace").strip()
err = stderr.read().decode(errors="replace").strip()
with open(r"C:\Users\ahernz\AppData\Local\Temp\opencode\hpc_md5.txt", "w", encoding="utf-8") as f:
    f.write(out + "\n---STDERR---\n" + err)
ssh.close()
print("written to hpc_md5.txt")
