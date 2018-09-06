from sklearn import datasets
import numpy as np

#original code
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
    
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, clf=knn)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn import metrics

#adjust k
from sklearn.metrics import accuracy_score
#try k=1 through k=25 and record testing accuracy
k_range=range(1,26)
scores=[]
for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)
    y_pred=knn.predict(X_test_std)
    scores.append(accuracy_score(y_test,y_pred))

k_optimal_index=scores.index(max(scores))
k_optimal=k_range[k_optimal_index]
K=list()
K.append(k_optimal)
for i in range(k_optimal,25):
    if scores[i]==max(scores):
        K.append(i+1)

print()
print('The best choice of k is/are', K)

for i in range(len(K)):
    print('When k = ',K[i])
    knn= KNeighborsClassifier(n_neighbors=K[i])
    knn.fit(X_train_std, y_train)
    y_pred=knn.predict(X_test_std)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    print(metrics.classification_report(y_test,y_pred,target_names=iris.target_names))
    
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, clf=knn)
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

print("My name is LI WENNING")
print("My NetID is: wenning5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
