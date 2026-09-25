import pandas as pd

df = pd.read_csv("data/processed/microsoft_training.csv")

print(df.head())
print("\nColumns:")
print(df.columns)

print("\nFailure Counts:")
print(df["failure"].value_counts())