from pathlib import Path
import pandas as pd

# Resolve dataset path relative to this script, not the current working directory
data_file = Path(__file__).resolve().parent.parent / 'Dataset' / 'ShipCase1.csv'
if not data_file.exists():
	raise FileNotFoundError(f"Dataset file not found: {data_file}")

df1 = pd.read_csv(data_file)
print(df1)