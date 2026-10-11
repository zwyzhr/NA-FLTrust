import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    output = root / 'bin' / ('nafltrust-ckks.exe' if os.name == 'nt' else 'nafltrust-ckks')
    output.parent.mkdir(exist_ok=True)
    subprocess.run(['go', 'build', '-o', str(output), '.'], cwd=root / 'crypto', check=True)


if __name__ == '__main__':
    main()
