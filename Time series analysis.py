from statsmodels.tsa.ar_model import AR
from random import random

data = [x + random() for x in range(1, 100)] # p-1 variables

model = AR(data)
model_fit = model.fit()

y_pred = model_fit.predict(len(data), len(data))
y_pred


y_pred is the next value i.e the Pth value of the autoregression
and the above number of variable is p-1 variables

libraries to handle the big data in python:-
sparkML
sparklib

hortonworks - application IDE to handlle the big data store the big data libraries and packages
official page