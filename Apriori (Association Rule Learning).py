import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from efficient_apriori import apriori

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
