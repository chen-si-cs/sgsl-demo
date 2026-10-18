"""Small interactive SSH/SFTP helper; credentials live only in process memory."""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("--user", default="chen")
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass("SSH password: ")
    client.connect(args.host, username=args.user, password=password,
                   look_for_keys=False, allow_agent=False, timeout=20)
    password = None
    print("CONNECTED", flush=True)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "run":
                    _, stdout, stderr = client.exec_command(request["command"], timeout=60)
                    output = stdout.read().decode("utf-8", "replace")
                    errors = stderr.read().decode("utf-8", "replace")
                    result = {"exit_code": stdout.channel.recv_exit_status(),
                              "stdout": output, "stderr": errors}
                elif op in {"put", "get"}:
                    with client.open_sftp() as sftp:
                        if op == "put":
                            sftp.put(request["local"], request["remote"])
                        else:
                            destination = Path(request["local"])
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            sftp.get(request["remote"], str(destination))
                    result = {"status": "ok", "op": op}
                elif op == "exit":
                    break
                else:
                    raise ValueError(f"Unsupported op: {op}")
                print(json.dumps(result, ensure_ascii=True), flush=True)
            except Exception as error:
                print(json.dumps({"error": str(error)}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
