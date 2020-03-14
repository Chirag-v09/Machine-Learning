# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:09:20 2020

@author: Chirag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Getting the dataset
df = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

# Making the dataset into the matrix form i.e list of list
transactions = []
for i in range(len(df)):
    transactions.append(df.iloc[i, :])

# Removing the nan values from the matrix
for i in range(len(df)):
    transactions[i] = [ item for item in transactions[i] if str(item) != 'nan']

# Getting the distinct items in the items
items = []
for i in range(len(df)):
    for j in range(len(df.iloc[i, :])):
        if df.iloc[i, j] not in items:
            items.append(df.iloc[i, j])

# Removing the nan value from the list
items = [item for item in items if str(item) != 'nan']

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
tra = lab.fit_transform(items)

lab.classes_

transactions_encoded = []
for i in range(len(transactions)):
    transactions_encoded.append(lab.transform(transactions[i]))

def table(dataset):
    result = np.zeros((len(dataset), 120))
    for i, value in enumerate(dataset):
        result[i, value] = 1
    return result


transactions_table = table(transactions_encoded)
df_transactions_table = pd.DataFrame(transactions_table)

items_freq = []
for i in range(df_transactions_table.shape[1]):
    val = 0
    for j in range(len(df_transactions_table)):
        val += df_transactions_table.iloc[j, i]
    items_freq.append(val)

avg = sum(items_freq)/120.0
data = {"items_freq": items_freq, "items": items}
df_data = pd.DataFrame(data)


def recommendation_by_item(item):
    num = lab.transform([item])
    item_transactions = df_transactions_table.iloc[:, num[0]]
    similar_items = df_transactions_table.corrwith(item_transactions)
    similar_items = similar_items.dropna()
    df = pd.DataFrame(similar_items)
    df = df.join(df_data)
    df = df.sort_values(['items_freq'], ascending=False)[:15]
    
    return df

item_name = "vegetables mix"
df = recommendation_by_item(item_name)
print(df["items"])
print(df)

















