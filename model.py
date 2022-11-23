import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random
np.random.seed(0) 

#4 month
x = list(range(1,121))

df = pd.read_csv('mldata.csv')

data = df.values.tolist()

y = []

for i in data:
    val = round(i[0]) + random.choice([1, 2, 3])
    y.append(val)

print(y)

train_x = x[:96]
train_y = y[:96]

test_x = x[96:]
test_y = y[96:]

# model
mymodel = np.poly1d(np.polyfit(train_x, train_y, 2))

# R squeared
print(r2_score(test_y, mymodel(test_x)))

# poly line
myline = np.linspace(1, 120, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# predict
print(mymodel(2))







    