# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:17:11 2020

@author: Chirag
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#####-----------

from apyori import apriori

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

# frq_items = apriori(dataset, min_support = 0.05, use_colnames = True) 

transactions_dirty = []
for i in range(len(dataset)):
    transactions_dirty.append(dataset.iloc[i, :])

transactions = []
for i in range(len(dataset)):
    transactions.append([item for item in transactions_dirty[i] if str(item) != 'nan'])


result = list(apriori(transactions))
result[0]



#####----------

from efficient_apriori import apriori, generate_rules_apriori

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

transactions_dirty = []
for i in range(len(dataset)):
    transactions_dirty.append(dataset.iloc[i, :])

transactions = []
for i in range(len(dataset)):
    transactions.append([item for item in transactions_dirty[i] if str(item) != 'nan'])

transactions_tuple = []
for i in range(len(dataset)):
    transactions_tuple.append(tuple(transactions[i]))

itemsets, rules = apriori(transactions_tuple, min_support = 0.05, min_confidence = 0.05)
print(rules)
# list(generate_rules_apriori(transactions_tuple, min_confidence = 0.1, num_transactions = 1))
# It needs dictionary