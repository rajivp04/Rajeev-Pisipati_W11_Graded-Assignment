from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import gzip

data = pd.read_csv('data/breast_cancer.csv')

data = data.set_index('id')
del data['Unnamed: 32']
data['diagnosis'] = data['diagnosis'].replace(['B', 'M'], [0, 1]) 

y = data.pop('diagnosis')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = []
estimators.append(('logistic', LogisticRegression()))
estimators.append(('cart', DecisionTreeClassifier()))
estimators.append(('svm', SVC()))

ensemble = VotingClassifier(estimators)

pipe = Pipeline([
    ('imputer', SimpleImputer()), 
    ('scaler', MinMaxScaler(feature_range=(0, 1))), 
    ('model', ensemble) 
])

pipe.fit(X_train, y_train)

print("Accuracy: %s" % str(pipe.score(X_test, y_test)))

print(ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test))
plt.show()

joblib.dump(pipe, gzip.open('model/model_binary.dat.gz', "wb"))
