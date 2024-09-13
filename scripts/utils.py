import pandas as pd

# Load the original emotions.csv
df = pd.read_csv('data/emotions.csv')

# Select the first 20 rows
df_sample = df.head(1000)

# Save the new emotions.csv
df_sample.to_csv('data/emotions.csv', index=False)
