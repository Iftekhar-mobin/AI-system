#https://miamioh.instructure.com/courses/38817/pages/dimensionality-reduction
#There are two common methods for PCA:
#Method 1: Covariance Matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


boston = load_boston()
#We load the features X and standardize it.

X = boston["data"]
X_std = StandardScaler().fit_transform(X)
#Now, create the covariance matrix between each variable using this data. Here the ‘.T’ means we’re transposing the matrix. 

covariance_matrix = np.cov(X_std.T)
#Next, store the eigenvalues and eigenvectors of this matrix in arrays using numpy’s linear algebra library, ‘linalg’.

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
#Finally, calculate the percentage of the variance that each variable explains using simple statistics. 

total_var = sum(eig_vals)
explained_var = [(i / total_var) for i in eig_vals]
print(explained_var)

plt.plot(explained_var)
plt.show()

#Method 2: Singular Value Decomposition (SVD)
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
pca.fit(X_std)
print(pca.explained_variance_ratio_)

#The example below uses the chi squared (chi^2) statistical test for non-negative features to select 4 of the best features from the Pima Indians onset of diabetes dataset.
# load data
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

url = "samples/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

# Recursive Feature Elimination
#The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.

# feature extraction
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d", fit.n_features_)
print("Selected Features: %s",fit.support_)
print("Feature Ranking: %s",fit.ranking_)


#Feature Importance
#Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

plt.plot(model.feature_importances_)
plt.show()
