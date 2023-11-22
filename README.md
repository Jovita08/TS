# TS
## Exp-1 Plot a time series data:
```
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("Dataset.csv",parse_dates=["Date"],index_col="Date")
df.head()
df["2022-01"]
df["2022-01"].Close.mean()
df["2022-04-01":"2022-07-31"]
df.Close.resample('M').mean()
mean=df.Close.resample('M').mean().plot(kind="bar")
mean=df.Close.resample('Y').mean().plot(kind="bar")
```
## Exp-2 Implementation of time series analysis and decomposition
```
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
df=pd.read_csv('AirPassengers.csv')
df.head()
df.set_index('Month',inplace=True)
df.index=pd.to_datetime(df.index)
df.dropna(inplace=True)
df.plot()
result=seasonal_decompose(df['#Passengers'], model='multiplicable',period=12)
result.seasonal.plot()
result.trend.plot()
result.plot()
```
## Exp-3 Implementation of log transformation on airline passenger
```
import numpy as np
import pandas as pd
data= pd.read_csv('AirPassengers.csv')
data.head()
data.dropna(inplace=True)
x=data['Month']
y=data['#Passengers']
data_log=np.log(data['#Passengers'])
X=data['Month']
Y=data_log
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.xlabel('Original Data')
plt.plot(X,Y)
plt.xlabel('Log- Transformed data')
```
## Exp-4 Compute auto correlation function
```
import matplotlib.pyplot as plt
import numpy as np
data = [3, 16, 156, 47, 246, 176, 233, 140, 130,
        101, 166, 201, 200, 116, 118, 247,
        209, 52, 153, 232, 128, 27, 192, 168, 208,
        187, 228, 86, 30, 151, 18, 254,
        76, 112, 67, 244, 179, 150, 89, 49, 83, 147, 90,
        33, 6, 158, 80, 35, 186, 127]
lags = range(35)
# Pre-allocate autocorrelation table
acorr = len(lags) * [0]
# Mean
mean = sum(data) / len(data)
# Variance
var = sum([(x - mean)**2 for x in data]) / len(data)
# Normalized data
ndata = [x - mean for x in data]
# Go through lag components one-by-one
for l in lags:
    c = 1 # Self correlation

    if (l > 0):
        tmp = [ndata[l:][i] * ndata[:-l][i]
            for i in range(len(data) - l)]

        c = sum(tmp) / len(data) / var
        #print(c)
        acorr[l] = c
print(acorr)
plt.grid(True)
plt.plot(acorr)
plt.title("Autocorrelation plot")
plt.xlabel("Lags")
plt.show()
```
## Exp-5 Auto regression model
```
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
df=pd.read_csv("rainfall.csv")
df
X=df['temp']
X
X.plot()
from statsmodels.tsa.stattools import adfuller
dtest=adfuller(X,autolag='AIC')
print("ADF:",dtest[0])
print("P value:",dtest[1])
print("No. of lags:",dtest[2])
print("No. of observations used for ADF regression:",dtest[3])
X_train=X[:len(X)-15]
X_test=X[len(X)-15:]
AR_model=AutoReg(X_train,lags=13).fit()
print(AR_model.summary())
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
pacf=plot_pacf(X,lags=25)
acf=plot_acf(X,lags=25)
pred=AR_model.predict(start=len(X_train),end=len(X_train)+len(X_test)-1,dynamic=False)
pred.plot()
X_test
pred
import sklearn.metrics
mse=sklearn.metrics.mean_squared_error(X_test,pred) 
mse**0.5
X_test.plot()
pred.plot()
```
## Exp-6 Moving average model in python
```
import os
os.chdir("C:\\users\\monit\\TIME SERIES LAB\\")
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import ExponentialSmoothing,SimpleExpSmoothing,Holt
import rcParams
rcParams['figure.figsize']= 20,5
import warnings
warnings.filterwarnings('ignore')
electricitytimeseries = pd.read_csv('Electric_Production.csv',header=0,index_col=0)
electricitytimeseries.shape
electricitytimeseries.head(20)

# MOVING AVERAGE METHOD

plt.plot(electricitytimeseries[1:50]['Value'])
plt.xticks(rotation=30)
plt.show()

# rolling average transfrom
rollingseries = electricitytimeseries[1:50].rolling(window=5)
rollingmean = rollingseries.mean()

# finding rolling mean MA(5)
print(rollingmean.head(10))

# plot transfrom dataset
rollingmean.plot(color='purple')
pyplot.show()

# rolling average transfrom
rollingseries = electricitytimeseries[1:50].rolling(window=10)
rollingmean = rollingseries.mean()
#finding rolling mean MA()
print(rollingmean.head(10))

# Exponential smoothing - single
data = electricitytimeseries[1:50]
fit1 = SimpleExpSmoothing(data).fit(smoothing_level=0.2,optimized=False)
fit2 = SimpleExpSmoothing(data).fit(smoothing_level=0.8,optimized=False)
plt.figure(figsize=(18,8))
plt.plot(electricitytimeseries[1:50],marker='o',color='black')
plt.xticks(rotation=30)
plt.plot(fit1.fittedvalues,marker='o',color='blue')
plt.plot(fit2.fittedvalues,marker='o',color='red')
```
## Exp-7 Implement ARMA in pythom
```
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [10, 7.5]

ar1 = np.array([1,0.33])
ma1 = np.array([1,0.9])
ARMA_1 = ArmaProcess(ar1,ma1).generate_sample(nsample = 1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
```
## Exp-8 Implementation of ARIMA model
```
import pandas as pd
df = pd.read_csv('website_data.csv')
df.info()
df.plot()
import numpy as np
df = np.log(df) # don't forget to transform the data back when making real predictions
df.plot()
msk = (df.index < len(df)-30)
df_train = df[msk].copy()
df_test = df[~msk].copy()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf_original = plot_acf(df_train)

pacf_original = plot_pacf(df_train)
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')
df_train_diff = df_train.diff().dropna()
df_train_diff.plot()
acf_diff = plot_acf(df_train_diff)

pacf_diff = plot_pacf(df_train_diff)
adf_test = adfuller(df_train_diff)
print(f'p-value: {adf_test[1]}')
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train, order=(2,1,0))
model_fit = model.fit()
print(model_fit.summary())
import matplotlib.pyplot as plt
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()
acf_res = plot_acf(residuals)

pacf_res = plot_pacf(residuals)
forecast_test = model_fit.forecast(len(df_test))

df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)

df.plot()
import pmdarima as pm
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=False)
auto_arima
auto_arima.summary()
forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
df['forecast_auto'] = [None]*len(df_train) + list(forecast_test_auto)

df.plot()
```
## Exp-9 Polynomial trend estimation
### a) linear tren estimation
```
# Function to calculate b
def calculateB(x, y, n):
    sx = sum(x)
    sy = sum(y) 
    sxsy = 0
    sx2 = 0
    for i in range(n):
        sxsy += x[i] * y[i]
        sx2 += x[i] * x[i]
    b = (n * sxsy - sx * sy)/(n * sx2 - sx * sx)
    return b

# Function to find the least regression line
def leastRegLine(X,Y,n):
    b = calculateB(X, Y, n)
    meanX = int(sum(X)/n)
    meanY = int(sum(Y)/n)
    a = meanY - b * meanX
    print("Linear Trend:")
    print("Y = ", '%.3f'%a, " + ", '%.3f'%b, "*X", sep="")

# Driver code
# Statistical data 
X = [95, 85, 80, 70, 60 ]
Y = [90, 80, 70, 65, 60 ]
n = len(X)
leastRegLine(X, Y, n)
```
### b) Polynomial regression
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('data.csv')
datas

X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color='blue')
plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='blue')
plt.plot(X, lin2.predict(poly.fit_transform(X)),color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

# Predicting a new result with Linear Regression after converting predict variable to 2D array
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)

# Predicting a new result with Polynomial Regression after converting predict variable to 2D array
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))
```
Exp-10 Holt winters method
```
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

import pandas as pd
airline  = pd.read_csv('AirPassengers.csv',index_col='Month',parse_dates=True)
airline.plot()
airline.freq = 'MS'
airline.index
len(airline)
train_airline = airline[:108] 
test_airline = airline[108:] 
fitted_model = ExponentialSmoothing(train_airline['#Passengers'],
                                  trend='mul',seasonal='mul',seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(36).rename('HW Test Forecast')
test_predictions[:10]

train_airline['#Passengers'].plot(legend=True,label='TRAIN')
test_airline['#Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
plt.title('Train and Test Data');

train_airline['#Passengers'].plot(legend=True,label='TRAIN')
test_airline['#Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters');
print("Mean Absolute Error = ",mean_absolute_error(test_airline,test_predictions))
final_model = ExponentialSmoothing(airline['#Passengers'],
                                  trend='mul',seasonal='mul',seasonal_periods=12).fit()
forecast_predictions = final_model.forecast(steps=36)

airline['#Passengers'].plot(figsize=(12,8),legend=True,label='Current Airline Passengers')
forecast_predictions.plot(legend=True,label='Forecasted Airline Passengers')
plt.title('Airline Passenger Forecast');
```
