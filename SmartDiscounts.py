import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from collections import OrderedDict
from pylab import rcParams
from sklearn import linear_model
from sklearn.model_selection import train_test_split
rcParams['figure.figsize'] = 14, 8


#Create data for Smart Discounts (helps for plotting purposes)
data = OrderedDict(
  amount_spent =  [50,  10, 20, 5,  95,  70,  100,  200, 0],
  send_discount = [0,   1,  1,  1,  0,   0,   0,    0,   1])

#Read in gender classification data 
data_2 = pd.read_csv('heights_weights_genders.csv')
A = data_2.drop('Gender', axis = 1)
B = data_2['Gender']

#Split gender classification data
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.2, random_state = 42)

#Replacement copy of target outputs for use in plotting
T = B.copy()
T = T.replace('Male', 0)
T = T.replace('Female', 1)

#Create x and Y values for sigmoid function to be plotted for comparison
x = np.linspace(-7,12,100)
Y = expit(x)

#Sort dataset into X and y values (inputs and targets). Reshape as appropriate
df = pd.DataFrame.from_dict(data)
X = df['amount_spent'].astype('float').values
X=X.reshape(9,1)
y = df['send_discount'].astype('float').values

#Plot smart_discounts data, gender classification and sigmoid
fig, axs = plt.subplots(3)
axs[0].scatter(X,y)
axs[1].scatter(A['Weight'],A['Height'], c = T, cmap = 'copper_r')
axs[2].plot(x,Y)
plt.show

#Create Logistic Rgeression model object
clf = linear_model.LogisticRegression() # Extra  Logistic Regression params: C=1e40, solver='newton-cg'

#Input new parameters
new_spend = int(input('How much did you spend today (to the nearest £): '))
height = int(input('How tall are you (to the nearest cm): '))
weight = int(input('How much do you weigh (to the nearest kilo): '))

#Fit and predict for smart discounts
fitted_model = clf.fit(X,y)
prediction_result = clf.predict([[new_spend]])

#Fit and predict for gender classification
fitted_model_2 = clf.fit(A_train,B_train)
prediction_result_2 = clf.predict([(weight,height)])

#Print predictions and accuracy of gender prediction
if prediction_result == 1.0:
    print(f" \nyou get the discount, and we are {100*(round(clf.score(A,B),2))}% you are a {prediction_result_2[0]}")
else:
     print(f" \nyou don't get the discount, and we are {100*(round(clf.score(A_test,B_test),2))}% you are a {prediction_result_2[0]}")
