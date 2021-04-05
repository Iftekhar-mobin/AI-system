#https://miamioh.instructure.com/courses/38817/pages/dimensionality-reduction
#There are two common methods for PCA:
#Method 1: Covariance Matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
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



#X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
#clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf.fit(X, y)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=2, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=0, verbose=0, warm_start=False)
#print(clf.feature_importances_)
