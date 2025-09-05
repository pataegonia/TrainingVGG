# run_all.py
import sys
import subprocess
from pathlib import Path

SCRIPTS = ["a.py", "b.py"]

def run_one(script: str):
    p = Path(__file__).with_name(script)
    if not p.exists():
        raise FileNotFoundError(f"{script} 파일을 찾을 수 없습니다: {p}")
    print(f"\n=== {script} 실행 시작 ===")
    # 각 스크립트를 현재 파이썬 인터프리터로 실행
    proc = subprocess.run([sys.executable, str(p)], cwd=p.parent, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{script} 실행 실패 (exit {proc.returncode})")

if __name__ == "__main__":
    for s in SCRIPTS:
        run_one(s)
    print("\n모든 스크립트 실행 완료.")
