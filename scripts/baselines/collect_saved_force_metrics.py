from pathlib import Path
import getpass
import json
import os
import shlex

import paramiko

root = Path(__file__).resolve().parents[2]
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
    remote_variables = ["FOG_REMOTE_RUN_ROOT", "FOG_REMOTE_SOURCE_ROOT"]
    if "FOG_REMOTE_REFERENCE_ROOT" in os.environ:
        remote_variables.append("FOG_REMOTE_REFERENCE_ROOT")
    remote_environment = " ".join(
        f"{name}={shlex.quote(os.environ[name])}" for name in remote_variables
    )
    stdin, stdout, stderr = client.exec_command(f"{remote_environment} python3 -", timeout=240)
    remote_script = Path(__file__).with_name("summarize_saved_forces_remote.py")
    stdin.write(remote_script.read_text(encoding="utf-8"))
    stdin.channel.shutdown_write()
    data = stdout.read().decode()
    error = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(error)
    value = json.loads(data)
    assert len(value) == 21
    output = root / "results/baselines_seed2026/force_metrics_seed2026.json"
    output.write_text(json.dumps(value, indent=2))
    print("Verified saved-force statistics collected for", len(value), "records. No simulation rerun.")
finally:
    client.close()
