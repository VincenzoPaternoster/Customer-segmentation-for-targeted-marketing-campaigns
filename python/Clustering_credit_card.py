# -*- coding: utf-8 -*-

"""
# Customer segmentation through clustering



"""
### Library
## Basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

## Sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## Setting
SEED=30
pd.set_option('display.float_format', '{:.4f}'.format)  # 4 digits

"""### 0. Dataset"""

## Dataset
df=pd.read_csv("https://raw.githubusercontent.com/VincenzoPaternoster/Customer-segmentation-for-targeted-marketing-campaigns/refs/heads/main/data/credit_card_customers.csv")
df.head()

"""### 1. Exploratory analysis

#### 1.1 Distribution analysis
"""

## View basic information about the dataset
df.info()
df.describe()

## See distributions of numeric columns

df.hist(bins=30,figsize=(15,19),edgecolor="black")
plt.tight_layout()
plt.show()

### NOTE: 1) Cash advance frequency should have values from 0 to 1
###       2) Generally distributions are asymmetric (skewed)

## CASH ADVANCE FREQUENCY column has values greater than 1
## Its values should be between 0 and 1

df[df["CASH_ADVANCE_FREQUENCY"]>1] # 8 values and 3 of them have also MINIMUM_PAYMENTS greater than PAYMENTS

## Exclude eight rows from analysis [681,1626,2555,2608,3038,3253,8055,8365]
df=df.drop(df[df["CASH_ADVANCE_FREQUENCY"]>1].index)

## Check
df[df["CASH_ADVANCE_FREQUENCY"]>1]

"""#### 1.2 Missing values"""

### See wich columns have missing values
df.isna().sum()

"""### 2) Preprocesing data

#### 2.1 Handling missing values
"""

### Replace NA in the columns MINIMUM_PAYMENTS and CREDIT_LIMIT with median
# (there are many outliers so I decided to use median)

## Use SimpleImputer() and replace NA with median
simp=SimpleImputer(strategy="median")
df[["MINIMUM_PAYMENTS","CREDIT_LIMIT"]]=simp.fit_transform(df[["MINIMUM_PAYMENTS","CREDIT_LIMIT"]])

## Check if there are another NA values
df.isna().sum()

## There are observation with MINIMUM_PAYMENTS greater than PAYMENTS
## Perhaps this anomaly describes risky customer behavior or errors in the data set.

df_cl=df.drop(df[df["MINIMUM_PAYMENTS"]>df["PAYMENTS"]].index) # 2361 of 8942 rows
                                                               # (approximately 26% of observations)

## Since there are a large number of customer accounts with minimum payments higher than the payments,
## I decided to divide the dataset into two subsets (with and without these values)
## in order to apply the clustering technique to both

"""##### *NOTE*:

**I created new dataset without outliers in order to analyze both datasets with and without them**

#### 2.2 Normalize numeric columns
"""

## New columns for payment habits with outliers

## PRC_MIN_PAYMENTS is the percentage of minimum payments divided by all payments
## It might be helpful to understand customers' payment habits
## It can take values between 0 and 1

df["PRC_MIN_PAYMENTS"]=np.where(df["PAYMENTS"]!=0,df["MINIMUM_PAYMENTS"]/df["PAYMENTS"],0)
df[df["PRC_MIN_PAYMENTS"]>1] #There are 2393 with PRC_MIN_PAYMENTS greater than 1

# Normalization with outliers

# Since I already have columns with distribution of values ​​from 0 to 1 and
# since the values ​​do not respect a normal distribution, I decided to normalize the values, not using standardization

df_norm=['BALANCE','PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE', 'PRC_MIN_PAYMENTS']

mm=MinMaxScaler()
df[df_norm]=mm.fit_transform(df[df_norm])
df.describe()

## New colums for payments habit without outliers
## It can take values between 0 and 1
df_cl["PRC_MIN_PAYMENTS"]=np.where(df_cl["PAYMENTS"]!=0,df_cl["MINIMUM_PAYMENTS"]/df_cl["PAYMENTS"],0)

df_cl[df_cl["PRC_MIN_PAYMENTS"]>1] # without outliers there are not PRC_MIN_PAYMENTS greater than 1

# Normalization without outliers
df_norm2=['BALANCE','PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE', 'PRC_MIN_PAYMENTS']

mm=MinMaxScaler()
df_cl[df_norm2]=mm.fit_transform(df_cl[df_norm2])
df_cl.describe()

"""#### 2.3 Correlation matrix"""

## Correlation matrix (what features should I use?)

# Select feature to correlate
df_corr=df.drop("CUST_ID",axis=1)

# Create correlation matrix
plt.figure(figsize=(14,10))
sbn.heatmap(df_corr.corr(),
            annot=True,
            xticklabels=df_corr.columns,
            yticklabels=df_corr.columns,
            fmt=".2f")

# The correlation matrix shows usual correlations between BALANCE and all operations
# that increase balance (e.g. increasing purchases, payments and type of payments (e.g. cash advance))
# I will use feautures to define clusters about:

# 1) Average expenses : Balance,One-off purchases and Installment purchases
# 2) Payment habits : Payments and Prc_Min_Payments
# 3) Frequency of use : Purchases frequency and Cash advance

"""### 3) Clustering, interpretation and marketing strategies

#### 3.1 Clustering : Preparation
"""

### Define function to choose number of clusters to use

def how_many_clusters(X,title=False):

    # Get intertias and silhouettes
    inertias = []
    silhouettes = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=SEED).fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))

    # Elbow method
    plt.plot(range(2, 11), inertias, marker='^')
    plt.xlabel("k")
    plt.ylabel("Inertia")

    if title:

      plt.title("Elbow Method with outliers")

    else:
      plt.title("Elbow Method without outliers")

    plt.show()

    # Silhouette score
    plt.plot(range(2, 11), silhouettes, marker='o', color='green')
    plt.xlabel("k")
    plt.ylabel("Score")

    if title:

      plt.title("Silhouette Score with outliers")

    else:
      plt.title("Silhouette Score without outliers")
    plt.show()

# View clusters of model

# Function for view clusters

def view_clust(X,k,title=False):

    # X features
    # k number of clusters

    # KMeans
    kmeans=KMeans(n_clusters=k,init="k-means++",random_state=SEED).fit(X)
    centers=kmeans.cluster_centers_
    y_kmeans=kmeans.predict(X)

    # View clusters

    # 2 FEATURES
    if X.shape[1]==2:

      # Set labels of axes
      plt.xlabel(X.columns[0])
      plt.ylabel(X.columns[1])

      # Plot scatter
      sbn.scatterplot(x=X[X.columns[0]],y=X[X.columns[1]],hue=pd.Categorical(y_kmeans),s=100)
      plt.scatter(centers[:,0],centers[:,1],c="black",s=200,alpha=0.8,marker="*")

      # Write SSD value on chart
      plt.text(1.2,0,f"SSD: {kmeans.inertia_:.2f}",ha='right', va='bottom', transform=plt.gca().transAxes)

      # Set title of chart
      if title:

        plt.title(f"Scatterplot of {X.columns[0]} and {X.columns[1]} with outliers")

      else:

        plt.title(f"Scatterplot of {X.columns[0]} and {X.columns[1]} without outliers")

      plt.legend(title="Cluster")
      plt.show()

    # 3 FEATURES
    elif X.shape[1] == 3:

        fig = plt.figure(figsize=(18, 8))

        views = [(20, 45), (30, -60)]  # 3D Views: I decided to use two graphs for the same data because
                                       #           I noticed that only some clusters were visible, so I decided
                                       #           to plot two graphs with two different perspectives to
                                       #           improve the subsequent interpretation.

        # Use a loop to plot two different views of the same graph

        for i, (elev, azim) in enumerate(views, start=1): ## elev= moves the angle from the Y to the Z
                                                          ## azim= moves the angle from the X to the Y

            ax = fig.add_subplot(1, 2, i, projection='3d') ## Set 1 row, 2 columns and the number of charts
                                                           ## to display two charts side by side

            ax.view_init(elev=elev, azim=azim) # set elevation and azim in degrees instead of radians
                                               # makes it easier to move the angle
            # Set name of axes
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.set_zlabel(X.columns[2])

            # Assign labels for each cluster
            for cluster in np.unique(y_kmeans): ## For each unique cluster of y_kmeans

                labs = y_kmeans == cluster ## create boolean mask to understand which observations belong to the current cluster

                ax.scatter3D(X.loc[labs, X.columns[0]], # to filter for each column or feature the observations associated to each cluster
                             X.loc[labs, X.columns[1]],
                             X.loc[labs, X.columns[2]],
                             label=f"Cluster {cluster}", s=60, alpha=0.7)

            # Show centroids
            ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2],
                         c='black', s=200, alpha=0.8, marker="*")

            # Show SSD value
            ax.text2D(0.05, 0.95, f"SSD={kmeans.inertia_:.2f}", transform=ax.transAxes)

            # Set title
            title_text = f"Scatterplot of {X.columns[0]}, {X.columns[1]}, {X.columns[2]}"

            if title:
                title_text += " with outliers"

            else:
                title_text += " without outliers"

            ax.set_title(title_text)

        plt.legend(title="Cluster", loc='upper right')
        plt.tight_layout()
        plt.show()

    else:
      print("Unable to display chart with this number of dimensions")

## Clustering models

# FIRST MODEL with outliers
avg_exp=df[["BALANCE","ONEOFF_PURCHASES","INSTALLMENTS_PURCHASES"]]

# FIRST MODEL without outliers
avg_exp2=df_cl[["BALANCE","ONEOFF_PURCHASES","INSTALLMENTS_PURCHASES"]]

# SECOND MODEL with outliers
habit_pay=df[["PAYMENTS","PRC_MIN_PAYMENTS"]]

# SECOND MODEL without outliers
habit_pay2=df_cl[["PAYMENTS","PRC_MIN_PAYMENTS"]]

# THIRD MODEL with outliers
freq_use=df[["CASH_ADVANCE","PURCHASES_FREQUENCY"]]

# THIRD MODEL without outliers
freq_use2=df_cl[["CASH_ADVANCE","PURCHASES_FREQUENCY"]]

"""#####

#### 3.2 Clustering : Models

#### **Note**

##### In the following lines of code, I decided to apply clustering to the dataset with outliers as well, because I want to show how these observations can influence cluster definition and subsequent marketing strategies.

##### 3.2.1 First model with outliers (Balance,One-off purchases,Installment purchases)
"""

## Number of clusters for the first model with outliers
how_many_clusters(avg_exp,True) ## k=4

# The elbow method suggests up to six/seven clusters, but the silhouette score is too low with six/seven.
# Therefore, the choice of four clusters represents the best compromise between the two methods.
# Although four clusters are not the optimal choice according to the elbow method alone.

## First clustering with outliers
view_clust(avg_exp,4,True)

"""
##### 3.2.2 First model without outliers (Balance,One-off purchases,Installment purchases)
"""

## Number of clusters for the first model without outliers

how_many_clusters(avg_exp2,False)

## First clustering without outliers
view_clust(avg_exp2,5,False)

"""
##### 3.2.3 Clustering: Second model with outliers (Payments and Prc_Min_Payments)
"""

## Number of clusters for the second clustering with outliers
how_many_clusters(habit_pay,True) ## k=5

## The best solution is 5 clusters
# Although 5 clusters are right after the elbow,
# the silhouette score for 5 clusters is good and looking at the
# arrangement of observations and clusters 5 seems optimal

## Second clustering with outliers

view_clust(habit_pay,5,True)

"""
##### 3.2.4 Second model without outliers (Payments and Prc_Min_Payments)
"""

## Number of clusters for the second clustering without outliers

how_many_clusters(habit_pay2,False)

## The elbow method suggests 6/7 clusters
## Instead silhouette score provide a good score ​​for 5 clusters
## I decided to use five clusters

## Second clustering without outliers

view_clust(habit_pay2,5,False)

## Unlike the previous graph, in this graph we can see how the outliers have influenced
## the results of previous work.
##Here we can see five well-defined clusters with all observations visible.

"""
##### 3.2.5 Third model with outliers (Purchases frequency and Cash advance)
"""

## Number of clusters for the third model with outliers
how_many_clusters(freq_use,True) ## k=5


## The best solution is five clusters
#  The elbow method shows that 6/7 clusters are the best number of cluster but
#  the silhouette score tend to reduce for 6/7 clusters.
#  For this reason I decide to use 5 clusters

## Third clustering with outliers
view_clust(freq_use,5,True)

"""
##### 3.2.6 Third model without outliers (Purchases frequency and cash advance)
"""

## Number of clusters for the third model without outliers

how_many_clusters(freq_use2,False)

# The elbow method shows that 6/7 clusters is the best solution but
# silhouette score shows a low value from 5 to 8 values compared to 4.
# Since the goal of the project is clustering I think that is useless put too many clusters
# for the segmentation.
# For this reason I decide to use 5 clusters

## Third clustering without outliers
view_clust(freq_use2,5,False)




In order to investigate these three characteristics, three models with and without outliers were used. Marketing strategies were provided for each model.
"""

