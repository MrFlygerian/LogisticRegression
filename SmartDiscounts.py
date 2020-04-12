import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from collections import OrderedDict

from sklearn import linear_model


#Create Logistic Rgeression model object
clf = linear_model.LogisticRegression() # Extra  Logistic Regression params: C=1e40, solver='newton-cg'


#Create data (this data set is tiny, this is only a training example)
data = OrderedDict(
  amount_spent =  [50,  10, 20, 5,  95,  70,  100,  200, 0],
  send_discount = [0,   1,  1,  1,  0,   0,   0,    0,   1])

#Create x and Y values for sigmoid function to be plotted for comparison
x = np.linspace(-7,12,100)
Y = expit(x)

#Sort dataset into X and y values (inputs and targets). Reshape as appropriate
df = pd.DataFrame.from_dict(data)
X = df['amount_spent'].astype('float').values
X=X.reshape(9,1)
y = df['send_discount'].astype('float').values

#Plot data with the sigmoid function for comparison
fig, axs = plt.subplots(2)
axs[0].scatter(X,y)
axs[1].plot(x,Y)
plt.show

#Fit model to the signmoid and use it to make a prediction
fitted_model = clf.fit(X,y)
new_spend = 90
prediction_result = clf.predict([[new_spend]])

#Show the predictions
if prediction_result == 0.0:
    print (f"a spend of £{new_spend} will get the discount")
else:
    print(f"a spend of £{new_spend} won't get the discount")
        