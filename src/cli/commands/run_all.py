import subprocess
import sys

def run(module: str):
    exe = sys.executable or "python"
    cmd = f'"{exe}" -m {module}'
    print(">>>", cmd, flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise SystemExit(f"Failed: {cmd}")

def main():
    run("src.data_ingest.fetch_bonds")
    run("src.data_ingest.process_bonds")
    run("src.data_ingest.map_poa_sa2")
    run("src.models.nowcast")
    run("src.models.forecast")
    run("src.reporting.validate_and_report")

if __name__ == '__main__':
    main()
