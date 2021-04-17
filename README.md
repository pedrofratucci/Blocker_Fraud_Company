# Transactions Fraud Prediction
![Main figure](https://sejaumdatascientist.com/wp-content/uploads/2020/10/corruption.png)


# Business Problem

In process...


# Objective and Solution Proposal

### Objective

Increase the Blocker Fraud Company (BF) profit. By increase the actual fraud transaction analysis recall, precision and stability.

### Solution Proposal

Create a binary classification toll with statics model and machine learning to increase the fraud diagnosis recall, precision and accuracy.


# Dataset Summary

- **step** - [int]: Maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation)
- **type** - [categorical]: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER
- **amount** - [float]: Amount of the transaction in local currency
- **nameOrig** - [string]: Customer who started the transaction
- **oldbalanceOrg** - [float]: Initial balance before the transaction
- **newbalanceOrig** - [float]: New balance after the transaction
- **nameDest** - [string]: Customer who is the recipient of the transaction
- **oldbalanceDest** - [float]: Initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants)
- **newbalanceDest** - [float]: New balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants)
- **isFraud** - [binary]: This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system
- **isFlaggedFraud** - [binary]: The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than \$ 200.000 in a single transaction

#  Mind Map Hypoteses
![mind_map](https://user-images.githubusercontent.com/81817799/115088913-8d49f380-9ee7-11eb-8ad6-efd978ca3d9a.png)


# Exploratory Data Analysis

## Univariate Analysis


### Categorical features distribution analysis

![cat_features_distribution](https://user-images.githubusercontent.com/81817799/115088928-9b980f80-9ee7-11eb-9bf2-4483fd6fc815.png)

- There are no transactions executed by Merchants
- Most of the transactions have account type customer as final destination
- Payment and Cash-Out as the most common type of transactions
- Transactions happens more often between day 6 and 17
- Transactions happens more often in the second month's week
- Transactions happens more during workweeks than weekends
- Transactions happens more often during the first month's fortnight
- There are more transactions with amounts between \$ 0.00 and \\$ 50,000.00 values than other amounts levels
- There is a almost percentage (close to 1%) of transactions which is a fraud, between all transactions executed


### Numerical features distribution analysis

![num_features_distribution](https://user-images.githubusercontent.com/81817799/115088964-ace11c00-9ee7-11eb-8d03-d93215fa3ec1.png)

All numerical variables have a large number of outliers yet, except the 'step' feature.


## Bivariate Analysis

### HYPOTESE 1: There are more fraudulent transactions in quantity through Cash-Out type

![H1_TF](https://user-images.githubusercontent.com/81817799/115089004-c6826380-9ee7-11eb-8053-48e28c3d3d33.png)
There are just Transfer and Cash-Out type in fraudulent transactions.
The Transfer type is the most common type between both, surpassing with a slightly difference of 3.


### HYPOTESE 2: There are more fraudulent transactions in total values through Transfer type

![H2_TF](https://user-images.githubusercontent.com/81817799/115089006-c84c2700-9ee7-11eb-96e7-6d08f0d90b26.png)
Transfer type also have more fraudulent transactions executed in total values then Cash-Out type.


### HYPOTESE 3: All transactions over $ 200,000.00 are fraudulents

![H3_TF](https://user-images.githubusercontent.com/81817799/115089010-c97d5400-9ee7-11eb-906c-bf6d9296a0b7.png)
Not all transactions over \$ 200,000.00 are fraudulent.
Only a small percentage of the transactions with \$ 200,000.00 amount or higher are fraudulent.


### HYPOTESE 4: Transactions with amount values level between $ 50,000.00 and 200,000.00 are more likely to be fraudulent than the others amount levels

![H4_TF](https://user-images.githubusercontent.com/81817799/115089012-ca15ea80-9ee7-11eb-9e46-f9e292b90c64.png)
Transaction level between $ 50,000.00 and \\$ 200,000.00 have more fraudulent transactions than between \\$ 0.00 and \\$ 50,000.00 level.
Transaction level with amounts higher than \$ 200,000.00 have more fraudulent transactions of all of them.


### HYPOTESE 5: Fraudulent transations happens more to Customer-Customer than Customer-Merchant relation

![H5_TF](https://user-images.githubusercontent.com/81817799/115089013-cb471780-9ee7-11eb-9b2b-9be4efdb9816.png)
There are only fraudulent transactions between Customer-Customer relation.


### HYPOTESE 6: There is more chance of having a fraudulent transaction when the final origin's balance is zero

![H6_TF](https://user-images.githubusercontent.com/81817799/115089017-cda97180-9ee7-11eb-8e17-bc9819964bd8.png)
Only a few percent of the fraudulent transactions leave the origin account's final balance with values different than zero.


### HYPOTESE 7: Fraudulent transactions tends to happen more on the weekends than workweek

![H7_TF](https://user-images.githubusercontent.com/81817799/115089018-cf733500-9ee7-11eb-868a-240f5afccdcd.png)
Fraudulent transactions are more likely to happen during the workweek than weekends.


### HYPOTESE 8: Fraudulent transactions tends to happen more on the First month's Fortnight than the Second month's Fortnight

![H8_TF](https://user-images.githubusercontent.com/81817799/115089020-d00bcb80-9ee7-11eb-8bfa-c91187990635.png)
Despite the fact that transactions are more likely to happen at the first month's fortnight, the fraudulent transactions are more likely to happen, with a slightly difference, at the second month's fortnight.


## Multivariate Analysis

![features_relations](https://user-images.githubusercontent.com/81817799/115089471-aa32f680-9ee8-11eb-8c33-8e1723953a1b.png)
**There are some variables that have considerable impact over 'is_fraud' result values:**

- 'amount'
- 'step'
- 'moth_fortnight'
- 'amount_level'
- 'flow_dest'
- 'is_flagged_fraud'

# Machine Learning Models Testing

To start, the following machine learning models were tested:

![machine_learning_models](https://user-images.githubusercontent.com/81817799/115126568-54bd1f00-9fa6-11eb-9b00-05682f0459fb.png)

- **On the business side:** We are aiming Precision score. Because each 5% increases on it is a R$ 500 increase in the diagnosis test price
- **On the patient side:** We are aiming Recall score. Because it reduces the chance of having a False Negative test result. As we know, false negatives results, frequently, comforts people into not retest. And in this case, there will be people with a growing disease that they don't know

So, we will choose the Top 4 best F1 Score models above to analyze, which is a metric that takes into account the Precision and the Recall metrics.

# Machine Learning Model Proposed

![final_machine_learning_model](https://user-images.githubusercontent.com/81817799/115125769-b7abb780-9fa0-11eb-9106-5fb2cea58791.png)

- All LGBM models have almost the same metrics, in general
- Taking in account that the LGBM Default model have a lower Recall and lower F1 score than the others, it will be cut out, thinking about the patients interests
- Taking in account that the LGBM Tuned & Calibrated model have a lower Precision and Recall intervals than the LGBM Tuned model, it will be cut out, thinking about the patients interests

So, **we will choose the LGBM Tuned as the final model to propose.**

**OBS:** For more about the decisions made and how it was done: [Blocker Fraud notebook](https://github.com/pedrofratucci/Blocker_Fraud_Company/blob/main/notebooks/blocker_fraud_PH.ipynb)


# Business Solution Performance

## Business accuracy methods comparison

In process...


## Business revenue methods comparison

In process...
  
  
# Further Improvements

- Develop an app that intakes a transaction and flags it's probability of being fraudulent or not

- Build a model retraining pipeline as the database of transactions grow


# References

## Business Problem Source
- https://sejaumdatascientist.com/crie-uma-solucao-para-fraudes-em-transacoes-financeiras-usando-machine-learning/

## Data Source
- https://www.kaggle.com/ntnu-testimon/paysim1
