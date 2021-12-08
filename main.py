import pandas as pd

df = pd.read_csv('datasets/telco-customer-churn.csv')

print(df.columns.values)

for col in df.columns.values:
    un = df[col].unique()
    print('{} -> {} {}'.format(col, len(un), un))

for col in df.columns.values:
    print(df[col].value_counts())

print(df.head())

one_hot_gender = pd.get_dummies(df['gender'], prefix='gender')
df = pd.concat([df, one_hot_gender], axis=1).drop('gender', axis=1)

print(df.head())
