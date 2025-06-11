# venv_path_check.py

import sys
import os

def main():
    venv_path = os.environ.get('VIRTUAL_ENV')

    if venv_path:
        print(f"✅ Virtual environment is active.")
        print(f"📁 VENV directory: {venv_path}")
    else:
        print("⚠️ No virtual environment is currently active.")

    print(f"\n🐍 Python executable in use: {sys.executable}")

if __name__ == "__main__":
    main()
