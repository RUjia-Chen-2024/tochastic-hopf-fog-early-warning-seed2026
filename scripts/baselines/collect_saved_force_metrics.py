from pathlib import Path
import getpass
import json
import os

import paramiko

root = Path(__file__).resolve().parents[1]
host = os.environ["FOG_SSH_HOST"]
username = os.environ["FOG_SSH_USER"]
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
try:
    client.connect(
        host,
        username=username,
        password=getpass.getpass("SSH password: "),
        timeout=15,
    )
    stdin, stdout, stderr = client.exec_command("python3 -", timeout=240)
    stdin.write((root / "scripts/summarize_saved_forces_remote.py").read_text())
    stdin.channel.shutdown_write()
    data = stdout.read().decode()
    error = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(error)
    value = json.loads(data)
    assert len(value) == 21
    output = root / "analysis/two_second_primary/force_metrics_seed2026.json"
    output.write_text(json.dumps(value, indent=2))
    print("Verified saved-force statistics collected for", len(value), "records. No simulation rerun.")
finally:
    client.close()
