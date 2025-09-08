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
    run("src.fetch_bonds")
    run("src.process_bonds")
    run("src.map_poa_sa2")
    run("src.model_nowcast")
    run("src.model_forecast")
    run("src.validate_and_report")

if __name__ == '__main__':
    main()
