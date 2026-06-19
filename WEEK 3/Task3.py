import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#loading dataset
df=pd.read_excel("Product-Sales-Region.xlsx")
#droping date cols
df=df.drop(["Date", "OrderDate", "DeliveryDate"], axis=1,errors='ignore')
#converting text to numeric
X=pd.get_dummies(df, drop_first=True)
print(X.head())
print(X.dtypes)
#scaling
scaler=StandardScaler()
scaled_data=scaler.fit_transform(X)

#PCA
pca=PCA(n_components=2)
pca_data=pca.fit_transform(scaled_data)
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

#Elbow Method
wcss = []
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i,random_state=42,n_init=10)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11),wcss,marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Silhouette Score
scores = []
for i in range(2, 11):
    kmeans=KMeans(n_clusters=i, random_state=42, n_init=10)
    labels=kmeans.fit_predict(pca_data)

    score=silhouette_score(pca_data, labels)
    scores.append(score)

plt.plot(range(2, 11),scores,marker="o")
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.show()

best_k = scores.index(max(scores)) + 2
print("Best Number of Clusters =", best_k)
#final Model
model=KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters=model.fit_predict(pca_data)
df["Cluster"]=clusters
#visualization
plt.figure(figsize=(8,6))
for c in range(best_k):
    temp=pca_data[clusters == c]

    plt.scatter(temp[:,0],temp[:,1],label=f"Cluster {c}")
plt.title("Customer Segmentation using PCA and K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
#cluster Summary
print(df.groupby("Cluster").mean(numeric_only=True))
#business Personas
persona_names={0:"Premium Customers",1:"Regular Customers",2:"Discount Seekers",3:"High Value Customers",4:"Occasional Buyers"}
df["Persona"]=df["Cluster"].map(persona_names)
print(df[["Cluster", "Persona"]].drop_duplicates())
#saving output
df.to_excel("WEEK 3/cluster_output.xlsx",index=False)
