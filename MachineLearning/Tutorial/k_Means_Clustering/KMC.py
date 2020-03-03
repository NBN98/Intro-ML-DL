import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)          #scaling down to comptute faster
y = digits.target

k = len(np.unique(y))               # oder k = 10
samples, features = data.shape      # Dimension

def bench_k_means(estimator, name, data):               #Implementierung des Estimators (Schätzer), Hier hat er verschiedene
                                                        #Parameter zum bewerten
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


clf = KMeans(n_clusters=k, init="random", n_init=10)      #init entspricht die Platzierung der Schwerpunkte
                                                            #n_init Anzahl der Schwerpunkte neu setzen
bench_k_means(clf, "1", data)