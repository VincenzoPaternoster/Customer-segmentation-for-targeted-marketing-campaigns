# 📌 Customer-segmentation-for-targeted-marketing-campaigns
A corporate has decided to segment its customers into homogeneous groups based on their spending behaviour and credit card usage.

This project analyzes a dataset provided by Profession AI during the Master's in Data Analysis with the aim of applying clustering techniques to obtain clusters based on customers' credit card usage.

---

## 📂 Repository Structure
```
E-commerce-sales-analysis/
│── data/
      ├── credit_card_customers.csv
│── images/
      ├── Distributions
      ├── Correlation matrix
      ├── Clustering
│── python/
      │──Clustering_credit_card.ipynb
      │──Clustering_credit_card.py
│── README.md
```

## 🎯 Project objectives
- 1. Perform clustering in order to segment the bank's customers based on the use of cards linked to:
  
      a) Average spending
      b) Payment habits
      c) Frequency of use

  ---

## 🗂️ Dataset
**Source:** Profession AI
**Period examined:** Undefined  
**Dimension of dataset:** 8950 customers

### 📌 Key variables
| Variables | Description |
|----------|-------------|
| Cust_id | Customer identifier |
| Balance | Residual balance on the purchase account |
| Balance_frequency | Balance update frequency (from 0 to 1) |
| Purchases| Total amount of purchases from the account|
| OneOffPurchases | Maximum amount for purchases in a single transaction |
| Installments_purchase | Amount of purchases made in installments|
| Cash advance | Cash advance given by the customer|
| Purchase_frequency | Frequency of purchases (from 0 to 1)|
| OneOffPurchasesFrequency | Frequency of purchases in a single transaction (from 0 to 1)|
| PurchasesInstallmentFrequency | Frequency of purchases made in installments (from 0 to 1)|
| CashAdvanceFrequency| Frequency with which a cash advance is requested (from 0 to 1)|
| CashAdvanceTRX| Number of cash transactions executed|
| CreditLimit| Maximum credit card limit for each customer|
| Payments| Total amount of payments executed by the customer|
| Minumum_Payments| Minimum payment due from the customer in a billing period, not necessarily the amount actually paid|
| PRC_Minimum_Payments| Ratio between the minimum payment due and the total amount paid by the customer|
| PRCFullPayment| Percentage of full paytment made by customer|
| Tenure| Length of credit card service for the customer (in years)|







---

## 🧹 Data Cleaning
Key operations performed:
- Handling missing values
- Handling inconsistent values (e.g. Minimum payments greater than Payments)
- Feature engineering (e.g. obtaining percent of minimum payments)
- Divide the dataset into two subsets based on the presence of outliers in order to highlight differences in clusters.
---

## 📊 Methodology:
- Used techniques (SimpleImputer,Normalization, Correlation matrix, K-means clustering)
- Main libraries (Pandas,Numpy,Sklearn,Matplotlib,Seaborn)


  Some of the distributions of characteristics were highly skewed and showed outliers (e.g. minimum_payments with values greater than payments) [See distributions](https://github.com/VincenzoPaternoster/Customer-segmentation-for-targeted-marketing-campaigns/blob/main/images/Distributions/Distributions.png)

  Since k-means clustering can be influenced by outliers and extreme values, I decided to split the dataset into two subsets. (with outliers [i.e. 26% of the original dataset] and without outliers)

  So I performed the elbow method, silhouette scoring, and clustering with both subsets, providing marketing strategies for each model. [With outliers](https://github.com/VincenzoPaternoster/Customer-segmentation-for-targeted-marketing-campaigns/tree/main/images/Clustering/With%20outliers) and [Without outliers](https://github.com/VincenzoPaternoster/Customer-segmentation-for-targeted-marketing-campaigns/tree/main/images/Clustering/Without%20outliers)

  
---

## 🔍 Key results
- Insight 1: There are anomalies in the columns relating to the frequency of cash advances and minimum payments. The reason for these errors must be verified.  [See outliers](https://github.com/VincenzoPaternoster/Customer-segmentation-for-targeted-marketing-campaigns/blob/main/images/Distributions/Distributions.png)
  
- Insight 2: There are differences in the number and shape of clusters between two datasets with and without outliers. In this case, the decision to split the dataset is related to the “educational purpose” (for example, it is not correct to normalize a dataset with outliers because normalization tends to reduce the effect of outliers, just as it is not correct to use k-means clustering when distributions are highly skewed).

- Insight 3: Clustering in the dataset without outliers showed well-defined clusters for the three models. Based on this, appropriate marketing strategies can be provided [See marketing strategies](https://github.com/VincenzoPaternoster/Customer-segmentation-for-targeted-marketing-campaigns/blob/main/python/Clustering_credit_card.ipynb)


---

## 🧠 Conclusions
- This project allowed me to use clustering techniques in the banking context with characteristics (variables) that can hide problems and errors that must be carefully verified.
- The project also helped me improve my knowledge of the banking context, which could be useful in future projects.
---

## 🛠️ Tools
- Python (pandas, numpy,sklearn,matplotlib, seaborn)
- Google Colab
- Obsidian

---

## 📬 Contacts

- **Vincenzo Paternoster**
- Email: vincenzopaternoster99@gmail.com
- LinkedIn: www.linkedin.com/in/vincenzo-paternoster
