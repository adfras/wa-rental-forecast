import subprocess

def run(cmd):
    print(">>>", cmd, flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise SystemExit(f"Failed: {cmd}")

def main():
    run("python -m src.fetch_bonds")
    run("python -m src.process_bonds")
    run("python -m src.map_poa_sa2")
    run("python -m src.model_nowcast")
    run("python -m src.model_forecast")
    run("python -m src.validate_and_report")

if __name__ == '__main__':
    main()
