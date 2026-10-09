"""Synthetic software fixtures, not real-world evaluation data."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 160
    age, bmi = rng.integers(20,85,n), rng.uniform(18,40,n)
    bp, glucose = rng.uniform(95,180,n), rng.uniform(65,220,n)
    cholesterol, smoker, activity = rng.uniform(120,280,n), rng.integers(0,2,n), rng.uniform(0,90,n)
    cardio = (age/85+bp/180+smoker*.3+cholesterol/280+rng.normal(0,.15,n)>2.3).astype(int)
    diabetes = (glucose/220+bmi/40+age/85+rng.normal(0,.1,n)>1.9).astype(int)
    pd.DataFrame(dict(age=age,bmi=bmi,systolic_bp=bp,glucose=glucose,cholesterol=cholesterol,smoker=smoker,activity_minutes=activity,cardiovascular=cardio,diabetes=diabetes)).to_csv(root / "health.csv",index=False)
    (root / "NOTICE.txt").write_text("SYNTHETIC SOFTWARE TEST FIXTURES. Not real-world evaluation data.", encoding="utf-8")
    return root

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    print(generate(parser.parse_args().destination))
