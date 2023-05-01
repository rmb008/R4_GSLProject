# This file reads in the field data from the Belovsky Ecosphere paper and runs a multiple linear
# regression with temp, salinity as predictors and micrograms of Chlorophytes per L as the response.
# Note: the zero-chlorophyte rows are removed for the MLR. This script also runs a zero-inflated
# Poisson regression model to handle the zeros.

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import linear_model
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels
#from statsmodels import statsmodels.discrete.discrete_model

#################################################################################################
# TODO LIST:

# - TODO: deal with negative fitted values
# - TODO: fit nonparametric lines (esp to diagnostic plots)
# - TODO: run diagnostic plots and test goodness of fit (esp for ZIP)
# - TODO: beautify figures
# - TODO: calculate residuals and/or error

#################################################################################################


#################################################################################################
# NOTE: Currently trying to solve convergence issue -  this is part of that

# deal with convergence warnings by ignoring for now - still gives different results every time, no good
#import warnings
#from statsmodels.tools.sm_exceptions import ConvergenceWarning
#warnings.simplefilter('ignore', ConvergenceWarning)
#################################################################################################

# Read in data
lmdata = pd.read_csv("field_data.csv")

# replace first header in csv (need second header to pull our needed columns)
new_header = lmdata.iloc[0]
lmdata = lmdata[1:]
lmdata.columns = new_header

# remove last row (it's the sum of all previous rows for each column)
lmdata = lmdata.drop(labels=157, axis=0)

# select the columns we need (note: the csv file has an extra space at the end of the temp string)
lmdata_reduced = lmdata[["MONTH", "YEAR", "SALINITY (%)", "TEMPERATURE (C) ", "CHLOROPHYTES (ug/l)"]]

# remove missing values (all denoted by 99999)
lmdata_filter1 = lmdata_reduced[lmdata_reduced["SALINITY (%)"] != '"99999"']
lmdata_filter2 = lmdata_filter1[lmdata_filter1["TEMPERATURE (C) "] != '"99999"']
lmdata_filter3 = lmdata_filter2[lmdata_filter2["CHLOROPHYTES (ug/l)"] != '"99999"']

# print(lmdata_filter3)
# print(lmdata_filter3.shape) # 58 observations of 3 variables is sufficient for regression

# need to re-index this for regression
lmdata = lmdata_filter3.reset_index()

# drop extra index column from re-indexing
lmdata = lmdata.drop(labels="index", axis=1)

# convert data from string to numeric
lmdata["SALINITY (%)"] = lmdata["SALINITY (%)"].astype(float)
lmdata["TEMPERATURE (C) "] = lmdata["TEMPERATURE (C) "].astype(float)
lmdata["CHLOROPHYTES (ug/l)"] = lmdata["CHLOROPHYTES (ug/l)"].astype(float)

#################################################################################################
# NOTE: This is for MLR model only and is not accurate but will be used for presentation!

# try dropping entries with zero chlorophytes:
#lmdata = lmdata[~(lmdata['CHLOROPHYTES (ug/l)'] == 0)]

# need to re-index this for regression
#lmdata = lmdata.reset_index()

# drop extra index column from re-indexing
#lmdata = lmdata.drop(labels="index", axis=1)
#################################################################################################

# assign predictor (X) and response (Y) variables for regression - note that X is an array
X = lmdata[["SALINITY (%)", "TEMPERATURE (C) "]]
Y = lmdata["CHLOROPHYTES (ug/l)"]

#################################################################################################
# NOTE: This is for MLR model only!!
# log transform the chlorophytes variable since it has a lot of small values near 0
#Y = np.log(Y) # note that this is the natural log
#################################################################################################

# split data into train and test sets:
mask = np.random.rand(len(lmdata)) < 0.5 # approx random 50/50 split (30 train, 28 test)
df_train = lmdata[mask]
df_test = lmdata[~mask]
#print('Training data set length='+str(len(df_train)))
#print('Testing data set length='+str(len(df_test)))
#print(df_train)
#print(df_test)
# set up regression expression in Patsy notation (for training set) - used for both models!
y_tr = df_train["CHLOROPHYTES (ug/l)"]
x1_tr = df_train["SALINITY (%)"]
x2_tr = df_train["TEMPERATURE (C) "]
expr_train = 'y_tr ~ x1_tr + x2_tr'
# and for testing set
y_te = df_test["CHLOROPHYTES (ug/l)"]
x1_te = df_test["SALINITY (%)"]
x2_te = df_test["TEMPERATURE (C) "]
expr_test = 'y_te ~ x1_te + x2_te'

# set up matrices for train and test sets
y_train, X_train = dmatrices(expr_train, df_train, return_type='dataframe')
y_test, X_test = dmatrices(expr_test, df_test, return_type='dataframe')
#print('y_train : ')
#print(y_train)
#print('')
#print('x_train : ')
#print(X_train)
#print('')
#print('y_test : ')
#print(y_test)
#print('')
#print('X_test : ')
#print(X_test)

#################################################################################################
# NOTE: MLR model starts here!!!
#################################################################################################

# run the regression using sklearn
regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train) # use just X,Y to get values from abstract (no data splitting) - but predictions in
                                   # abstract are still viable (this is just better)
fitted = model.predict(X_train) # these are the chlorophyte values predicted from/fitted to the regression model
# print(fitted)

# predict new values (23.5% salinity, 28.5 deg temp) -> this is a 2.5 deg C and 7.5% NaCl increase
new_x = {'Intercept': [1], 'x1_tr': [23.5], 'x2_tr': [28.5]}
new_x = pd.DataFrame(data=new_x)
y_pred = model.predict(new_x)
#print("MLR prediction:")
#print(y_pred)

# print parameters/coefficients
#print("MLR coefficients")
#print(regr.coef_)

# print intercept of fit line
#print("MLR Intercept")
#print(regr.intercept_)

# print R^2
#print("MLR R^2")
#print(model.score(X, Y)) # R^2 is about 0.44 for MLR with zeros removed

# plot X vs Y (these are just values from the actual data)
#plt.scatter(X["SALINITY (%)"], Y) # salinity and chlorophytes
#plt.show()
#plt.scatter(X["TEMPERATURE (C) "], Y) # temp and chlorophytes
#plt.show()

# plot X vs fitted (shows relation between predicted chloro values and predictors)
#plt.scatter(X["SALINITY (%)"], fitted)
#plt.show()
#plt.scatter(X["TEMPERATURE (C) "], fitted)
#plt.show()

# plot actual vs fitted (should be tightly fit around line y=x)
#plt.scatter(Y, fitted)
#plt.show()

# plot temp vs salinity (proxy test for correlation, shows these are related as we know in theory
                        # - should not affect prediction, just inference)
#plt.scatter(X["SALINITY (%)"], X["TEMPERATURE (C) "])
#plt.xlabel("Salinity (%)")
#plt.ylabel("Temperature (C)")
#plt.show()

# NOTE: See Rachel's code for 3D MLR plot(s)!!!

#################################################################################################
# NOTE: ZIP model begins here!
#################################################################################################

# do zero-inflated Poisson (ZIP) model (rate model with chlorophytes as rate):

#################################################################################################
# NOTE: trying to deal with convergence/ZIP fit problem by checking if not enough zero inflation

# check if zero inflation is real - currently doesn't work without statsmodels version 0.14 in dev
#mod = statsmodels.discrete.discrete_model.Poisson.from_formula(expr_train, df_train)
#res = mod.fit()
#diag = res.get_diagnostic()
#diag.test_poisson_zeroinflation().pvalue
#################################################################################################

# Run ZIP model
zip_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=X_train, exog_infl=X_train, inflation='logit').fit()
print(zip_training_results.summary())

# calculuate root mean square error (it's really root mean squared prediction error)
zip_predictions = zip_training_results.predict(X_test,exog_infl=X_test)
predicted_counts=np.round(zip_predictions)
actual_counts = y_test["y_te"]
print('ZIP RMSE='+str(np.sqrt(np.sum(np.power(np.subtract(predicted_counts,actual_counts),2)))))

# prediction plot
fig = plt.figure()
fig.suptitle('Predicted versus actual counts using the ZIP model')
predicted, = plt.plot(X_test.index, predicted_counts, 'go-', label='Predicted')
actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual')
plt.legend(handles=[predicted, actual])
plt.show()

# manually run predictions for particular values (note that param's 3 through 6 indices are intercept, salinity, temp)
# since ZIP is not converging, run this 3 times and take the average of the outputs
pred_chloro = np.exp(np.matmul([1, 23.5, 28.5], zip_training_results.params[3:6]))
print("ZIP prediction")
print(pred_chloro)

#################################################################################################
# NOTE: This is an attempt to get the predict method to work for the ZIP model

#pred_data = {'col1': [23.5], 'col2': [28.5]}
#pred_df = pd.DataFrame(data=pred_data)
#print(zip_training_results)
#print(pred_df.size)
#print(pred_df)
#pred_df = np.squeeze(np.asarray(pred_df))
#print(type(pred_df))
#print(type(pred_df["col1"]))
#specific_pred = zip_training_results.predict(pred_df, exog_infl=pred_df)
#future_preds = np.round(specific_pred)
#print(future_preds)
#################################################################################################

# TODO: check sample size (ok to have only 36 records?)

# TODO: run diagnostic plots

# TODO: 3D plots of MLR

# TODO: predicting, calculuate MSPE, etc

#TODO: calculate errors/residuals (distance from plane)
