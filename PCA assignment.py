
# Import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Import the dataset
wine= pd.read_csv("wine.csv")
wine.shape
wine.describe()
del wine["Type"]
wine

# Dividing the data into predictors and target values
x= wine.iloc[:,1:]
y= wine.iloc[:,0]

# Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)
X

# Importing PCA fron sklearn
from sklearn.decomposition import PCA

# Obtaining first 3 Principle components using PCA technique
pca= PCA(n_components=3)

pca_values= pca.fit_transform(X)
pca_values

# Variance ratios of first 3 principle components
var= pca.explained_variance_ratio_
var

var1= np.cumsum(np.round(var, decimals=4)*100)            # Variance ratios of first 3 principle components
var1

plt.plot(var1, c="r")
# With first 3 Principle components, 66.5 % of the information from dataset is retained

# Converting the array into dataframe
pca_data= pd.DataFrame(data=pca_values, index=range(0,178), columns=["PC1", "PC2","PC3"])

#HIERARCHIAL CLUSTERING
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# Using the "complete" linkage and "euclidean" method for dendrogram
z=linkage(pca_data, method="complete", metric="euclidean")
plt.figure(figsize=(18,8));plt.title("hei clus denrogram");plt.xlabel("index"); plt.ylabel("distance")
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=10.,
    )
# The Dendrogram shows 4 clear and disctinctive clusters at distance of 7

from sklearn.cluster import AgglomerativeClustering
h_complete= AgglomerativeClustering(n_clusters=4, linkage="complete", affinity="euclidean").fit(pca_data)

# Cluster labels
cluster_labels= pd.Series(h_complete.labels_)
cluster_labels

# Combining the labels with the dataset
wine['clust']= cluster_labels
wine.head()

# Getting the aggregate of each cluster mean
wine.iloc[:,1:].groupby(wine.clust).mean()

# Observations
# Cluster 0 is the wine type with lowest content of Alcohol, Proline and Color
# Cluster 3 is the wine type with highest content of Alcohol, Magnesium, Proline and Color
# Cluster 1 and 2 lies in between 0 and 3

#KMEANS CLUSTERING
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Plotting scree plot or elbow curve for k
k= list(range(2,10))
k

TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pca_data)
    WSS= []
    for j in range(i):
        WSS.append(sum(cdist(pca_data.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1, pca_data.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k, TWSS,'ro-');plt.xlabel("No of clusters");plt.ylabel("Total within SS");plt.xticks(k)


# The elbow can be found at k=3, after which the TWSS is decreasing gradually
# Building KMeans clustering model for k=3
model= KMeans(n_clusters=3)
model.fit(pca_data)

# Cluster labels
md= pd.Series(model.labels_)
md

# Combining the cluster labels with the dataset
wine['kclust']=md
wine.head()

# Aggregating each cluster by its mean
wine.iloc[:,1:14].groupby(wine.clust).mean()

# Observations
# Cluster 2 is the wine type with lowest content of Alcohol, Malic, Ash, Magnesium, Proline and Color
# Cluster 1 is the wine type with highest content of Alcohol, Ash, Magnesium, Phenols, and Proline
# Cluster 0 lies in between 2 and 1
