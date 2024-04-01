# COVID-19-Tracker

## Description
  A COVID-19 tracker utilizing machine learning for visualization and projection of confirmed, deceased, and recovered cases. This utilizes NumPy, Pandas, Matplotlib, Scikit-learn libraries to visualize and predict COVID-19 cases after n number of days. Shown are for code for sample confirmed cases. The following code can be configured to display either the three types of case, as well as changing country data.

## Code Description
### Data Cleaning and Transformation
  Read and transformed into a 1-dimensional array to create the training data from the CSV file from the data repository of the 2019 Novel Coronavirus Visual Dashboard operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cases
raw = pd.read_csv("time_series_covid19_confirmed_global.csv")
dates = raw.iloc[1, 4:].T
dates = dates.index
Tdates = np.arange(0,len(dates))
Tdates

PHdata = raw.iloc[182, 4:].T
```

### Scikit-learn modelling
  Shown are the generated linear-model analysis and polynomial-model analysis from the training data.
```python
from sklearn import linear_model

X = Tdates
y = PHdata.values

X = X.reshape(-1, 1)

mymodel = linear_model.LinearRegression().fit(X, y)

print("slope =", mymodel.coef_)
print("intercept =", mymodel.intercept_)

x1 = np.linspace(Tdates.min(), Tdates.max(), 100)
y1 = mymodel.predict(x1.reshape(-1, 1)).flatten() 
```

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = Tdates
y = PHdata.values

# preprocess
X    = X.reshape(-1, 1)
poly = PolynomialFeatures(degree = 4)
X    = poly.fit_transform(X)

mymodel2 = linear_model.LinearRegression()
mymodel2.fit(X, y)

print("slope =", mymodel2.coef_)
print("intercept =", mymodel2.intercept_)
```
### Scikit-learn prediction and data visualization
  Utilizing trained model from Sklearn for 4th degree polynomial regression analysis to predict COVID-19 case after n number of days (28 days). Generated model closely follows trained data which were used for case prediction.
  
```python
pred = mymodel2.predict(poly.fit_transform([[len(dates)+28]]))
pred

x = np.linspace(Tdates.min(), Tdates.max(), 100)
x = poly.fit_transform(x.reshape(-1, 1))
y = mymodel2.predict(x).flatten() 

fig, ax = plt.subplots(figsize=(20,8));
ax.plot(x[:,1], y, 'bo-', label='Polynomial Line')

ax.plot(x1, y1, 'y*', label='Linear Regression Line')

ax.scatter(dates, PHdata, label='Training Data', marker='$❤$', s = 200, c = 'r');

plt.xticks(dates, dates, rotation='vertical')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
ax.set_title('Confirmed Cases in Philippines vs. Date')
plt.grid('on')
plt.savefig('Philippines_Confirmed Cases.png')
```
### Sample Output
![Philippines_Confirmed Cases](https://github.com/kerbs001/COVID-19-Tracker/assets/155122597/298c9db6-b002-4f71-9d4f-f8cd539d1283)
