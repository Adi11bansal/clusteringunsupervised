import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot');
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
 
X=np.array([[1,2],
            [1.5,1.8],
            [5,8],
            [80,80],
            [1,0.6],
            [9,11]])

#plt.scatter(X[:,0],X[:,1],s=150,linewidths=5)
#plt.show()

distrotions=[]
K=range(1,6)
for k in K:
    clf=KMeans(n_clusters=k).fit(X)
    clf.fit(X)
    distrotions.append(sum(np.min(cdist(X,clf.cluster_centers_,'euclidean'),axis=1))/X.shape[0])
    

plt.plot(K,distrotions,'bx-')
plt.xlabel('k')
plt.ylabel('dis')
plt.show()
#centroids=clf.cluster_centers_
#labels=clf.labels_
#colors =10*["g.","r.","c.","b.","k."]

#for i in range(len(X)):
    #plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=15)
#plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5)
#plt.show()

