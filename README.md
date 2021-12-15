# ZS--Mekktronix-Sales-Forecasting

Problem Statement was taken from a challenge on Hackerearth (ZS Data Science Challenge - 2018)

Problem Statement:                                                                                                                         
Mekktronix is a global premium electronics manufacturing companies provides electronic devices across globe through their large distributor channel. Recently company is observing lot of fluctuations in their demand forecasting across geographies affecting their revenue. The company has reached out to build a machine learning driven forecasting solution to predict sales accurately.

Thus, we need to build a model that can accurately forecast the sales such that the company can cope up with the demands more easily.


Approach Followed:                                                                                                                         
Considered the data as a time series data with seasonality, with the presence of exogenous variables (SARIMAX model).    

The dataset had presence of multiple countries, and the Sales of multiple Products, hence needed to build separate models for every country and products.

Selected the best model, for various iterations over a range of p,d,q values, which had the lowest BIC.
