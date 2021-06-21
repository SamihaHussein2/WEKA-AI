from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#%matplotlib inline


path= "SeoulBikeData.csv"
df = pd.read_csv(path,encoding="latin1")
print(df.head())

plt.scatter(df.RentedBikeCount,df['Hour'])
plt.xlabel('Date')
plt.ylabel('Hour')

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['RentedBikeCount','Hour']])
y_predicted

df['cluster']=y_predicted
print(df.head())

km.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Date,df1['Hour'],color='green')
plt.scatter(df2.Date,df2['Hour'],color='red')
plt.scatter(df3.Date,df3['Hour'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('RentedBikeCount')
plt.ylabel('Hour')
plt.legend()


scaler = MinMaxScaler()

scaler.fit(df[['Hour']])
df['Hour'] = scaler.transform(df[['Hour']])

scaler.fit(df[['RentedBikeCount']])
df['RentedBikeCount'] = scaler.transform(df[['RentedBikeCount']])

plt.scatter(df.RentedBikeCount,df['Hour'])

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['RentedBikeCount','Hour']])
print(y_predicted)

df['cluster']=y_predicted
df.head()

km.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.RentedBikeCount,df1['Hour'],color='green')
plt.scatter(df2.RentedBikeCount,df2['Hour'],color='red')
plt.scatter(df3.RentedBikeCount,df3['Hour'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['RentedBikeCount','Hour']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
