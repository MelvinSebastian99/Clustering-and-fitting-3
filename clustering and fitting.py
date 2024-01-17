#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://data.worldbank.org/indicator/FI.RES.TOTL.CD


# In[64]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import sklearn.metrics as skmet
from sklearn.cluster import KMeans
import scipy.optimize as opt
import numpy as np
import errors


# In[3]:


def worldbank_data(filename):
    """
    Imports World Bank data.

    Returns:
    - Original Data.
    - Transposed Data.
    """    
    df = pd.read_csv(filename, skiprows=4)
    df_T = df.set_index(df.columns[0]).transpose()
    
    return df, df_T


# In[8]:


df, df_T = worldbank_data('API_FI.RES.TOTL.CD_DS2_en_csv_v2_6298254.csv')


# In[9]:


df.head(3)


# In[21]:


columns_to_select = ['Country Name', 'Indicator Name'] + list(map(str, range(2000, 2023)))
df_sub = pd.DataFrame()
for column in columns_to_select:
    df_sub[column] = df[column]
df_sub = df_sub.dropna()
df_subx = pd.DataFrame()
df_subx["Country Name"] = df_sub["Country Name"]
df_subx["2022"] = df_sub["2022"].copy()


# In[22]:


df_subx["Growth Since 2000 (%)"] = 100.0 * (df["2022"] - df["2000"]) / (df["2000"])
df_subx.head()


# In[23]:


df_subx.describe()


# In[15]:


plt.figure(figsize=(10, 6))
scatter_plot = plt.scatter(df_subx["2022"], df_subx["Growth Since 2000 (%)"], 10, label="Countries", c='black')
plt.xlabel("Total reserves (includes gold, current US$) in 2022")
plt.ylabel("Growth Since 2000 (%)")
plt.title("Different Countries with Total reserves")
plt.legend()
plt.show()


# In[35]:


def data_norm(data_frame, features):
    """
    Function to normalize the specified features using StandardScaler.

    Returns:
    - Normalized DataFrame.
    """
    scaler = RobustScaler()
    subset_features = data_frame[features]
    scaler.fit(subset_features)
    normalized_data = scaler.transform(subset_features)
    normalized_df = pd.DataFrame(normalized_data, columns=features)

    return normalized_df, scaler


# In[36]:


df_norm, scaler = data_norm(df_subx, ['2022', 'Growth Since 2000 (%)'])


# In[37]:


df_norm.head()


# In[38]:


def silhouette_score(xy, n):
    """
    Calculates silhouette score for n clusters using KMeans++ initialization.

    Returns:
    Silhouette score.
    """
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[39]:


for i in range(2, 10):
    score = silhouette_score(df_norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[46]:


kmeans = KMeans(n_clusters=2, init='k-means++', n_init=20)
kmeans.fit(df_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]


# In[49]:


plt.figure(figsize=(7, 5))
sns.scatterplot(x=df_subx["2022"], y=df_subx["Growth Since 2000 (%)"], hue=labels, palette="Set1", marker="o", s=10)
plt.scatter(xkmeans, ykmeans, marker="x", c="black", s=50, label="Cluster Centroids")
plt.xlabel("Total reserves (includes gold, current US$) in 2022")
plt.ylabel("Growth Since 2000 (%)")
plt.title("Different Countries with Total reserves")
plt.legend()
plt.show()


# In[55]:


df_uk = df_T.loc['2000':'2022', ['United Kingdom']].reset_index()
df_uk = df_uk.rename(columns={'index': 'Year', 'United Kingdom': 'UK Reserves'})
df_uk = df_uk.apply(pd.to_numeric, errors='coerce')
df_uk.describe()


# In[58]:


plt.figure(figsize=(10, 6))
sns.lineplot(data = df_uk, x='Year', y='UK Reserves', color='black')
plt.xlabel('Year')
plt.ylabel('UK Total Reserves')
plt.title('Total Reserves in th UK between 2000-2022')
plt.show()


# In[59]:


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    # makes it easier to get a guess for initial parameters
    t = t - 2010
    f = n0 * np.exp(g*t)
    return f


# In[61]:


param, covar = opt.curve_fit(exponential, df_uk["Year"], df_uk["UK Reserves"], p0=(1e11, 0.1))
df_uk["fit"] = exponential(df_uk["Year"], *param)


# In[63]:


plt.figure(figsize=(10, 6))
sns.lineplot(data=df_uk, x="Year", y="UK Reserves", label="UK Reserves", color='black')
sns.lineplot(data=df_uk, x="Year", y="fit", label="Exp Prediction", color='red')
plt.xlabel('Year')
plt.ylabel('UK Total Reserves')
plt.title('Total Reserves in th UK between 2000-2022')
plt.legend()
plt.show()


# In[67]:


import errors
years = np.arange(2022, 2034, 1)
predictions = exponential(years, *param)
confidence_range = errors.error_prop(years, exponential, param, covar)


# In[72]:


plt.figure(figsize=(10, 6))
sns.lineplot(x= df_uk["Year"], y= df_uk["UK Reserves"], label="UK Reserves", color='black')
sns.lineplot(x=years, y=predictions, label="Prediction", color='red')
plt.fill_between(years, predictions - confidence_range, predictions + confidence_range, 
                 color='red', alpha=0.4, label="Confidence Range")
plt.xlabel('Year')
plt.ylabel('UK Total Reserves')
plt.title('Total Reserves in th UK Prediction')
plt.legend()
plt.show()


# In[ ]:




