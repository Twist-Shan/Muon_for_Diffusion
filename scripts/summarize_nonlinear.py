"""Recreate the nonlinear analysis from archived measurements without training."""
import os
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND'] = 'Agg'
target = root / 'runs/nonlinear_summary'
if target.exists():
    raise SystemExit('Output already exists: runs/nonlinear_summary; preserve or rename it first.')
shutil.copytree(root / 'results/nonlinear', target)
sys.path.insert(0, str(root / 'experiments/nonlinear/frozen'))
from summarize import summarize
print(summarize(target))
