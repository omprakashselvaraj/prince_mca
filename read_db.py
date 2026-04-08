import duckdb
import pandas as pd


con = duckdb.connect('medical_codes.db')


df = con.execute("SELECT * FROM codes").df()

print("DataFrame shape:", df.shape)
print("DataFrame columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

con.close()

print("\nDatabase read as DataFrame successfully.")