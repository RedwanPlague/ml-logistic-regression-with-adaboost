import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif


def process_features(data, features_to_ignore, numerical_features, binary_features, categorical_features):
    for col in features_to_ignore:
        data = data.drop(col, axis=1)

    mean_imputer = SimpleImputer(strategy='mean')
    for col in numerical_features:
        if data[col].isnull().sum() > 0:
            data[col] = mean_imputer.fit_transform(data[[col]]).ravel()

    frequent_imputer = SimpleImputer(strategy='most_frequent')
    for col in categorical_features:
        if data[col].isnull().sum() > 0:
            data[col] = frequent_imputer.fit_transform(data[[col]]).ravel()

    for col in binary_features:
        data[col].replace({'Yes': 1, 'No': 0}, inplace=True)

    for col in categorical_features:
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col, prefix_sep='=')], axis=1)
        data = data.drop(col, axis=1)

    return data


def separate_labels(data, label_col_name):
    x = data.drop(label_col_name, axis=1)
    y = np.array(data[label_col_name]).reshape(-1, 1)
    return x, y


def select_on_info_gain(x_train, y_train, x_test, use_count=None):
    if use_count is None:
        print('using all features')
        return np.array(x_train), np.array(x_test)

    info_gain = mutual_info_classif(x_train, y_train.ravel(), random_state=0)
    gain_col_list = [(g, c) for g, c in zip(info_gain, x_train.columns)]
    gain_col_list.sort(reverse=True)
    final_features = [c for g, c in gain_col_list[:use_count]]
    print('used features: {}'.format(final_features))
    return np.array(x_train[final_features]), np.array(x_test[final_features])


def telco_customer_churn_preprocessor(use_count=None):
    df = pd.read_csv('datasets/telco-customer-churn.csv', na_values=' ')

    features_to_ignore = ['customerID']
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    binary_features = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    categorical_features = ['gender', 'Contract', 'PaymentMethod', 'InternetService', 'MultipleLines',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                            'StreamingMovies']

    # df.replace({'No phone service': 'No', 'No internet service': 'No'}, inplace=True)

    df['TotalCharges'] = df['TotalCharges'].astype('float')

    df = process_features(df, features_to_ignore, numerical_features, binary_features, categorical_features)

    x, y = separate_labels(df, 'Churn')

    min_max_scaler = MinMaxScaler()
    x[x.columns] = min_max_scaler.fit_transform(x[x.columns])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    x_train, x_test = select_on_info_gain(x_train, y_train, x_test, use_count)

    return x_train.T, x_test.T, y_train.T, y_test.T


def adult_data_preprocessor(use_count=None):
    train_df = pd.read_csv('datasets/adult/adult.data', header=None, na_values='?', sep=', ', engine='python')
    test_df = pd.read_csv('datasets/adult/adult.test', skiprows=1, header=None, na_values='?', sep=', ',
                          engine='python')

    features = ['age', 'work-class', 'final-weight', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                '>50K']
    features_to_ignore = []
    numerical_features = ['age', 'final-weight', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    binary_features = ['>50K']
    categorical_features = ['work-class', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']

    train_df.columns = features
    test_df.columns = features

    train_df['>50K'].replace({'>50K': 'Yes', '<=50K': 'No'}, inplace=True)
    test_df['>50K'].replace({'>50K.': 'Yes', '<=50K.': 'No'}, inplace=True)

    split_index, _ = train_df.shape
    df = train_df.merge(test_df, how='outer')

    df = process_features(df, features_to_ignore, numerical_features, binary_features, categorical_features)

    min_max_scaler = MinMaxScaler()
    df[df.columns] = min_max_scaler.fit_transform(df[df.columns])

    train_df, test_df = df.iloc[:split_index, :], df.iloc[split_index:, :]

    x_train, y_train = separate_labels(train_df, '>50K')
    x_test, y_test = separate_labels(test_df, '>50K')

    x_train, x_test = select_on_info_gain(x_train, y_train, x_test, use_count)

    return x_train.T, x_test.T, y_train.T, y_test.T


def accuracy(y_true, y_pred):
    return 100 * accuracy_score(y_true.ravel(), y_pred.ravel())


def add_column_of_ones(x):
    _, m = x.shape
    return np.r_[np.ones((1, m)), x]


class LogisticRegressor:
    def __init__(self, alpha=0.01, error_cutoff=0.0, max_iterations=1000, show_plot=False):
        self.alpha = alpha
        self.error_cutoff = error_cutoff
        self.max_iterations = max_iterations
        self.show_plot = show_plot
        self.w = None

    def fit(self, x, y):
        x = add_column_of_ones(x)
        y = np.where(y < 0.5, -1, 1)
        n, m = x.shape
        # self.w = np.random.rand(n, 1)
        self.w = np.zeros((n, 1))

        print_interval = self.max_iterations // 10

        errors = []
        for i in range(self.max_iterations):
            z = self.w.T @ x
            a = np.tanh(z)
            error = np.mean(np.square(y - a))
            if self.show_plot:
                errors.append(error.item())
            if i % print_interval == 0:
                print('({})'.format(i), end=' ')
            if error <= self.error_cutoff:
                break
            da = y - a
            dz = 1 - np.square(a)
            self.w += self.alpha * (x @ (da * dz).T) / m
        print()

        if self.show_plot:
            plt.plot(range(len(errors)), errors)
            plt.ylim(0)
            plt.show()

    def predict(self, x):
        x = add_column_of_ones(x)
        z = self.w.T @ x
        a = np.tanh(z)
        return a > 0.0


class AdaBoost:
    def __init__(self, models):
        self.k = len(models)
        self.h = models
        self.z = np.zeros(self.k)

    def fit(self, features, labels):
        n, m = features.shape
        w = np.ones((1, m)) / m

        for i in range(self.k):
            indices = np.random.choice(m, m, p=w.ravel())
            x_train, y_train = features[:, indices], labels[:, indices]
            self.h[i].fit(x_train, y_train)
            y_pred = self.h[i].predict(features)
            print('{:.2f}'.format(accuracy(labels, y_pred)))
            error = w[labels != y_pred].sum()
            if error > 0.5:
                continue
            w[labels == y_pred] *= error / (1 - error)
            w /= w.sum()
            self.z[i] = np.log((1 - error) / error)

    def predict(self, data):
        y_pos_neg_list = np.concatenate([np.where(h.predict(data), 1, -1) for h in self.h])
        y_pos_neg = self.z @ y_pos_neg_list
        return y_pos_neg > 0.0


def main():
    x_train, x_test, y_train, y_test = telco_customer_churn_preprocessor()
    # x_train, x_test, y_train, y_test = adult_data_preprocessor()

    # ab = AdaBoost([LogisticRegressor(alpha=0.1, max_iterations=2000, error_cutoff=0.0) for _ in range(10)])
    # ab.fit(x_train, y_train)
    # train_pred = ab.predict(x_train)
    # test_pred = ab.predict(x_test)

    lr = LogisticRegressor(alpha=0.1, max_iterations=2000, error_cutoff=0.0)
    lr.fit(x_train, y_train)
    train_pred = lr.predict(x_train)
    test_pred = lr.predict(x_test)

    print('Train Accuracy: {:.2f}'.format(accuracy(y_train, train_pred)))
    print('Test  Accuracy: {:.2f}'.format(accuracy(y_test, test_pred)))
    print('Train Precision: {:.2f}'.format(precision_score(y_train.ravel(), train_pred.ravel())))
    print('Test  Precision: {:.2f}'.format(precision_score(y_test.ravel(), test_pred.ravel())))
    print('Train Recall: {:.2f}'.format(recall_score(y_train.ravel(), train_pred.ravel())))
    print('Test  Recall: {:.2f}'.format(recall_score(y_test.ravel(), test_pred.ravel())))
    print('Train F1 Score: {:.2f}'.format(f1_score(y_train.ravel(), train_pred.ravel())))
    print('Test  F1 Score: {:.2f}'.format(f1_score(y_test.ravel(), test_pred.ravel())))


if __name__ == '__main__':
    main()
