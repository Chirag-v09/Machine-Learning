import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Library for ASR
from efficient_apriori import apriori

# Importing Dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None) # Dataset is available in Datasets Repository.

# Getting the list of items for each Transactions
transactions_dirty = []
for i in range(len(dataset)):
    transactions_dirty.append(dataset.iloc[i, :])

# Removing the 'nan' values from the Transactions
transactions = []
for i in range(len(dataset)):
    transactions.append([item for item in transactions_dirty[i] if str(item) != 'nan'])

# Making the list of lists into the list of tuples
transactions_tuple = []
for i in range(len(dataset)):
    transactions_tuple.append(tuple(transactions[i]))

# Making the rules through Apriori Lirary and printing it
itemsets, rules = apriori(transactions_tuple, min_support = 0.05, min_confidence = 0.05)
print(rules)
