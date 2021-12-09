import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def telco_customer_churn_preprocessor():
    df = pd.read_csv('datasets/telco-customer-churn.csv')

    features_to_ignore = ['customerID']
    numerical_features_to_impute = ['TotalCharges']
    categorical_features_to_impute = []
    binary_features = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'PaperlessBilling', 'Churn']
    categorical_features = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
    # features_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

    df.replace(r'^\s*$', np.NAN, regex=True, inplace=True)
    df.replace({'No phone service': 'No', 'No internet service': 'No'}, inplace=True)

    df['TotalCharges'] = df['TotalCharges'].astype('float')

    for col in features_to_ignore:
        df = df.drop(col, axis=1)

    mean_imputer = SimpleImputer(strategy='mean')
    for col in numerical_features_to_impute:
        df[col] = mean_imputer.fit_transform(df[[col]]).ravel()

    frequent_imputer = SimpleImputer(strategy='most_frequent')
    for col in categorical_features_to_impute:
        df[col] = frequent_imputer.fit_transform(df[[col]]).ravel()

    for col in binary_features:
        df[col].replace({'Yes': 1, 'No': 0}, inplace=True)

    for col in categorical_features:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, prefix_sep='=')], axis=1).drop(col, axis=1)

    min_max_scaler = MinMaxScaler()
    df = min_max_scaler.fit_transform(df)

    return np.array(df)


print(telco_customer_churn_preprocessor())
