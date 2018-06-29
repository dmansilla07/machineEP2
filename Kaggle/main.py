import numpy as np
import pandas as pd
from scipy import stats, integrate
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def normal_equation_prediction(X, y):
    """
    Calculates the prediction using the normal equation method.
    You should add a new column with 1s.

    :param X: design matrix
    :type X: np.array
    :param y: regression targets
    :type y: np.array
    :return: prediction
    :rtype: np.array
    """
    A = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
    inverse = np.linalg.pinv(np.matmul(np.transpose(A), A))
    prediction = np.matmul(np.matmul(inverse, np.transpose(A)),y)
    return np.matmul(A,(prediction))



label_encoder = LabelEncoder()

train_data = pd.read_csv('train.csv')
categorical_vars = train_data.describe(include=["object"]).columns
continous_vars = train_data.describe().columns
test_data = pd.read_csv('test.csv')
predictors = ['age', 'job', 'loan']



y_train = train_data['y']

tree_model = DecisionTreeClassifier()

X_train = train_data[predictors]


X_test = test_data[predictors]



"""
X_train.job = label_encoder.fit_transform(X_train.job)
X_train.loan = label_encoder.fit_transform(X_train.loan)

X_test.job = label_encoder.fit_transform(X_test.job)
X_test.loan = label_encoder.fit_transform(X_test.loan)


tree_model.fit(X_train, y_train)

y_test = tree_model.predict(X_test)

output = pd.DataFrame({'Id': test_data.id, 'y': y_test})
output.to_csv('output.csv', index=False)
"""

sns.distplot(train_data['age'], kde=False, fit=stats.gamma);

#print(train_data.describe(include=["object"]))
#print (train_data.groupby('pdays')['id'].nunique())
