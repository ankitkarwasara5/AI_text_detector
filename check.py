import pandas as pd

df = pd.read_csv('Dataset/train_essays.csv')
print("Data Distribution:")
# Use normalize=True to see percentages
print(df['generated'].value_counts(normalize=True))