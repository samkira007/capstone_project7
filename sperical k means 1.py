#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_blobs


# In[28]:


plt.figure(figsize=(6,5))


# In[32]:


n_samples =1500
random_state = 170
X,y = make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.60834549,-0.63667341],[-0.40887718,0.85253229]]
X_aniso= np.dot(X,transformation)
y_pred=KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
y_pred2=SpectralClustering(n_clusters=3,gamma=5,random_state=random_state).fit


# In[30]:


plt.scatter(X_aniso[:, 0],X_aniso[:, 1], c=y_pred2,s=10)
plt.title("spectral clustering")


# In[31]:


plt.scatter(X_aniso[:, 0],X_aniso[:, 1], c=y_pred,s=10)
plt.title("Kmeans clustering")


# In[ ]:





# In[ ]:





# In[ ]:




