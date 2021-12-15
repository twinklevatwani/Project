## Manish Nanwani
##ZS_Hackathon-HackerEarth-Mekktronix Sales Forecasting.

## Importing the required packages
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import itertools

import copy
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

pd.options.display.float_format = '{:20,.2f}'.format
pd.options.display.max_rows = 1000

## Reading the csv files to a pandas dataframe
train = pd.read_csv("F:\\ZS Hackathon\\dataset\\yds_train2018.csv")
holidays = pd.read_excel("F:\\ZS Hackathon\\dataset\\holidays.xlsx")
promo_expense = pd.read_csv("F:\\ZS Hackathon\\dataset\\promotional_expense.csv")
test = pd.read_csv("F:\\ZS Hackathon\\dataset\\yds_test2018.csv")

##Creating a new dataframe for transactions at month level, for every product_ID, every month every year, every country
train1 = pd.DataFrame(
    train.groupby(by=['Country', 'Product_ID', 'Year', 'Month'], as_index=False).agg({'Sales': 'sum'}))
train1['year_month'] = [str(x) + str('{0:02}'.format(y)) for x, y in zip(train1.Year, train1.Month)]
train1['date'] = [str(x) + '-' + str('{0:02}'.format(y)) for x, y in zip(train1.Year, train1.Month)]

##re-creating the table2, having the training horizons, ie, no of training points at month resolution
table2 = train1.groupby(by=['Country', 'Product_ID'], as_index=False).agg({'year_month': ['first', 'last', 'count']})
table2.columns = ['Country', 'Product_ID', 'From', 'To', 'Training Horizon']

##processing the holidays df, to count the number of holidays for every month,every year, every country
holidays['year'] = [int(d.split(',')[0]) for d in holidays.Date.tolist()]
holidays['month'] = [int(d.split(',')[1]) for d in holidays.Date.tolist()]
holidays1 = holidays.groupby(by=['Country', 'year', 'month'], as_index=False).agg({'Holiday': 'count'})

## merging the holidays info with the train dataframe
train_main = pd.merge(train1, holidays1, how='left', left_on=['Year', 'Month', 'Country'],
                      right_on=['year', 'month', 'Country'])

## merging the expenses info with the train dataframe
train_final = pd.merge(train_main, promo_expense, how='left', left_on=['Year', 'Month', 'Country', 'Product_ID'],
                       right_on=['Year', 'Month', 'Country', 'Product_Type'])

train_final.drop(['year', 'month', 'Product_Type'], axis=1, inplace=True)

## na values for holiday counts means there are no holidays for that month, so replace nan by 0.
train_final['Holiday'].fillna(0, inplace=True)
## Similarly, na values for expenses means there were no expense for that month on that product, so replacing it by 0
train_final['Expense_Price'].fillna(0, inplace=True)

train_final['Holiday'] = train_final.Holiday.astype('int64')
train_final.set_index('date', inplace=True)
# train_final.index = pd.to_datetime(train_final.index)

train_act = train_final[['Country', 'Product_ID', 'Holiday', 'Expense_Price', 'Sales']]
train_act.index = pd.to_datetime(train_act.index)

## Transformimg the test data as well
test1 = copy.deepcopy(test)
test1['year_month'] = [str(x) + str('{0:02}'.format(y)) for x, y in zip(test1.Year, test1.Month)]
test1['date'] = [str(x) + '-' + str('{0:02}'.format(y)) for x, y in zip(test1.Year, test1.Month)]

##re-creating the table5, having the testing horizons, ie, no of test points at month resolution
table5 = (test1.groupby(by=['Country', 'Product_ID'], as_index=False).agg({'year_month': ['first', 'last', 'count']}))
table5.columns = ['Country', 'Product_ID', 'From', 'To', 'Forecast Horizon']

## merging the holidays info with the test dataframe
test_main = pd.merge(test1, holidays1, how='left', left_on=['Year', 'Month', 'Country'],
                     right_on=['year', 'month', 'Country'])

## merging the expenses info with the test dataframe
test_final = pd.merge(test_main, promo_expense, how='left', left_on=['Year', 'Month', 'Country', 'Product_ID'],
                      right_on=['Year', 'Month', 'Country', 'Product_Type'])

##re-creating the table6, having the testing horizons, ie, no of test points at month resolution
table6 = copy.deepcopy(table5)
table6['From'] = table2['From']
table6.drop('Forecast Horizon', axis=1, inplace=True)
temp_df = test_final.groupby(by=['Country', 'Product_ID'], as_index=False).agg({'Expense_Price': 'count'})
table6['Expense Data'] = ['Available' if x else 'Not available' for x in temp_df.Expense_Price]

test_final.drop(['year', 'month', 'Product_Type'], axis=1, inplace=True)

## na values for holiday counts means there are no holidays for that month, so replace nan by 0.
test_final['Holiday'].fillna(0, inplace=True)
## Similarly, na values for expenses means there were no expense for that month on that product, so replacing it by 0
test_final['Expense_Price'].fillna(0, inplace=True)

test_final['Holiday'] = test_final.Holiday.astype('int64')
test_final.set_index('date', inplace=True)
# test_final.index = pd.to_datetime(test_final.index)
test_act = test_final[['Country', 'Product_ID', 'Holiday', 'Expense_Price', 'Sales']]
test_act.index = pd.to_datetime(test_act.index)

test2 = copy.deepcopy(test)

### Need to forecast for every country and every product ID
for c in train_act.Country.unique():
    for p in train_act.loc[train_act.Country == c, 'Product_ID'].unique():
        print("Starting with:", c, p)
        temp_train = train_act.loc[
            (train_act.Country == c) & (train_act.Product_ID == p), ['Holiday', 'Expense_Price', 'Sales']]
        temp_train.index = pd.DatetimeIndex(temp_train.index.values, freq=temp_train.index.inferred_freq)
        temp_test = test_act.loc[
            (test_act.Country == c) & (test_act.Product_ID == p), ['Holiday', 'Expense_Price', 'Sales']]
        temp_test.index = pd.DatetimeIndex(temp_test.index.values, freq=temp_test.index.inferred_freq)

        ## Sarimax
        p_val = range(0, 5)
        d_val = range(0, 5)
        q_val = range(0, 5)

        pdq = list(itertools.product(p_val, d_val, q_val))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p_val, d_val, q_val))]

        warnings.filterwarnings("ignore")

        best_aic = np.inf
        best_pdq = None
        best_seasonal_pdq = None
        tmp_model = None
        best_mdl = None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    tmp_mdl = sm.tsa.statespace.SARIMAX(endog=temp_train.Sales,
                                                        exog=temp_train[['Expense_Price', 'Holiday']],
                                                        order=param, seasonal_order=param_seasonal,
                                                        enforce_stationarity=False, enforce_invertibility=False)

                    res = tmp_mdl.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_pdq = param
                        best_seasonal_pdq = param_seasonal
                        best_mdl = tmp_mdl

                        # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, res.aic))
                except:
                    continue
        # print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))


        # define SARIMAX model and fit it to the data
        model = sm.tsa.statespace.SARIMAX(endog=temp_train.Sales, exog=temp_train[['Expense_Price', 'Holiday']],
                                          order=best_pdq, seasonal_order=best_seasonal_pdq,
                                          enforce_stationarity=True, enforce_invertibility=True)
        result = model.fit()

        ## forecast predictions
        pred = result.get_prediction(start=pd.to_datetime(temp_test.index[0]),
                                     end=pd.to_datetime(temp_test.index[-1]),
                                     exog=temp_test[['Expense_Price', 'Holiday']],
                                     dynamic=False)

        forecasts = abs(pred.predicted_mean)
        test2.loc[(test2.Country == c) & (test2.Product_ID == p), 'Sales'] = [x for x in forecasts]

test3 = copy.deepcopy(test2)
test3.index = test3.S_No
test3.drop('S_No', inplace=True, axis=1)
test3.to_csv("F:\\ZS Hackathon\\dataset\\submission2_manish.csv")