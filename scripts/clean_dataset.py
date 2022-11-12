import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Rename folders')
parser.add_argument('--source', type=Path, help='original dataset')
parser.add_argument('--target', type=Path, help='where to save new dataset')

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_excel(args.source)
    df = df[["ID_Paziente", "MVI", "EdmondsonGrading"]]
    df = df[(df["MVI"] == 0) + (df["MVI"] == 1)]
    df = df.fillna(0).replace(' ', 0)
    df["EdmondsonGrading"] = df["EdmondsonGrading"].astype(int)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.target)