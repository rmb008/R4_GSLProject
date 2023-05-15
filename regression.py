# Reads in the field data from the Belovsky Ecosphere paper and runs a
# multiple linear regression and zero inflated Poisson regression with temp,
# salinity as predictors and micrograms of Chlorophytes per L as the response.

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import linear_model
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels
import math
import random

# Read in data
lmdata = pd.read_csv("field_data.csv")

# replace first header in csv (need second header to pull our needed columns)
new_header = lmdata.iloc[0]
lmdata = lmdata[1:]
lmdata.columns = new_header

# remove last row (it's the sum of all previous rows for each column)
lmdata = lmdata.drop(labels=157, axis=0)

# select the columns we need (note: the csv file has an extra space at the end
# of the temp string)
lmdata_reduced = lmdata[["MONTH", "YEAR", "LAKE VOLUME (m^3)", "SALINITY (%)",
                         "TEMPERATURE (C) ", "CHLOROPHYTES (ug/l)"]]

# remove missing values (all denoted by 99999)
lmdata_filter1 = lmdata_reduced[lmdata_reduced["SALINITY (%)"] != '"99999"']
lmdata_filter2 = lmdata_filter1[lmdata_filter1["TEMPERATURE (C) "] != '"99999"']  # nopep8
lmdata_filter3 = lmdata_filter2[lmdata_filter2["CHLOROPHYTES (ug/l)"] != '"99999"']  # nopep8

# At this point, there are 58 observations.

# need to re-index this for regression
lmdata = lmdata_filter3.reset_index()

# drop extra index column from re-indexing
lmdata = lmdata.drop(labels="index", axis=1)

# convert data from string to numeric
lmdata["SALINITY (%)"] = lmdata["SALINITY (%)"].astype(float)
lmdata["TEMPERATURE (C) "] = lmdata["TEMPERATURE (C) "].astype(float)
lmdata["CHLOROPHYTES (ug/l)"] = lmdata["CHLOROPHYTES (ug/l)"].astype(float)

# assign predictor (X) and response (Y) variables for regression-
# note that X is an array
X = lmdata[["SALINITY (%)", "TEMPERATURE (C) "]]
Y = lmdata["CHLOROPHYTES (ug/l)"]

# set seed to control for randomness
random.seed(7)

# split data into train and test sets:
df_train = lmdata.sample(frac=0.5, random_state=123)
df_test = lmdata.drop(df_train.index)
# set up regression expression in Patsy notation (for training set)-
# used for both models!
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

##############################################################################
# NOTE: MLR model starts here!!!
##############################################################################

# run the regression using sklearn
regr = linear_model.LinearRegression()
# model = regr.fit(X_train, y_train)  # used to run training set
model = regr.fit(X, Y)  # use to run whole dataset
# fitted = model.predict(X_train)  # used to run training set
fitted = model.predict(X)  # whole dataset values

# Predict chlorophyte concentration from temp, salinity (23.5% salinity,
# 28.5 deg temp) -> this is a 2.5 deg C and 7.5% NaCl increase, uses
# training set
# new_x = {'Intercept': [1], 'x1_tr': [23.5], 'x2_tr': [28.5]}
# new_x = pd.DataFrame(data=new_x)
# y_pred = model.predict(new_x)
# print("MLR prediction:")
# print(y_pred)

# print p values (uses whole dataset)
params = np.append(regr.intercept_, regr.coef_)
predictions = regr.predict(X)
new_X = np.append(np.ones((len(X), 1)), X, axis=1)
M_S_E = (sum((Y-predictions)**2))/(len(new_X)-len(new_X[0]))
v_b = M_S_E*(np.linalg.inv(np.dot(new_X.T, new_X)).diagonal())
s_b = np.sqrt(v_b)
t_b = params / s_b
p_val = [2*(1-sp.stats.t.cdf(np.abs(i),
            (len(new_X)-len(new_X[0])))) for i in t_b]
p_val = np.round(p_val, 3)
print("MLR p values")
print(p_val)

# print parameters/coefficients
print("MLR coefficients")
print(regr.coef_)

# print intercept of fit line
print("MLR Intercept")
print(regr.intercept_)

# print R^2
print("MLR R^2")
print(model.score(X, Y))

# compute root mean squared prediction error - uses training/testing sets
# X_test = X_test.rename(columns={"x1_te": "x1_tr", "x2_te": "x2_tr"})
# mlr_predictions = model.predict(X_test)
# predicted_counts_mlr=np.round(mlr_predictions)
# actual_counts_mlr = y_test["y_te"]
# actual_counts_mlr = actual_counts_mlr.to_frame()
# print('MLR RMSE='+str(np.sqrt(np.sum(np.power(np.subtract(
#                                               predicted_counts_mlr,
#                                               actual_counts_mlr),2)))))

# plot X vs Y (these are just values from the actual data)
a, b = np.polyfit(X["SALINITY (%)"], Y, 1)
c, d = np.polyfit(X["TEMPERATURE (C) "], Y, 1)
plt.scatter(X["SALINITY (%)"], Y)  # salinity and chlorophytes
plt.plot(X["SALINITY (%)"], a*X["SALINITY (%)"]+b)
plt.title("Chlorophyte Concentration (Actual) vs. Salinity")
plt.xlabel("Salinity (% NaCl)")
plt.ylabel("Chlorophyte Concentration ($\mu$g/L)")
plt.show()
plt.scatter(X["TEMPERATURE (C) "], Y)  # temp and chlorophytes
plt.plot(X["TEMPERATURE (C) "], c*X["TEMPERATURE (C) "]+d)
plt.title("Chlorophyte Concentration (Actual) vs. Temperature")
plt.xlabel("Temperature ($^\circ$C)")
plt.ylabel("Chlorophyte Concentration ($\mu$g/L)")
plt.show()

# plot X vs fitted (shows relation between predicted chlorophytes, predictors)
e, f = np.polyfit(X["SALINITY (%)"], fitted, 1)
g, h = np.polyfit(X["TEMPERATURE (C) "], fitted, 1)
plt.plot(X["SALINITY (%)"], e*X["SALINITY (%)"]+f)
plt.title("Chlorophyte Concentration (Fitted) vs. Salinity")
plt.xlabel("Salinity (% NaCl)")
plt.ylabel("Chlorophyte Concentration ($\mu$g/L)")
plt.scatter(X["SALINITY (%)"], fitted)
plt.show()
plt.scatter(X["TEMPERATURE (C) "], fitted)
plt.plot(X["TEMPERATURE (C) "], g*X["TEMPERATURE (C) "]+h)
plt.title("Chlorophyte Concentration (Fitted) vs. Temperature")
plt.xlabel("Temperature ($^\circ$C)")
plt.ylabel("Chlorophyte Concentration ($\mu$g/L)")
plt.show()

# plot actual vs fitted (should be tightly fit around line y=x)
plt.scatter(Y, fitted)
plt.plot(Y, Y)
plt.title("Fitted vs. Actual Chlorophyte Concentration")
plt.xlabel("Actual Chlorophyte Concentration ($\mu$g/L)")
plt.ylabel("Fitted Chlorophyte Concentration ($\mu$g/L)")
plt.show()

# plot temp vs salinity
m, n = np.polyfit(X["SALINITY (%)"], X["TEMPERATURE (C) "], 1)
plt.scatter(X["SALINITY (%)"], X["TEMPERATURE (C) "])
plt.plot(X["SALINITY (%)"], m*X["SALINITY (%)"]+n)
plt.title("Temperature vs. Salinity")
plt.xlabel("Salinity (% NaCl)")
plt.ylabel("Temperature ($^\circ$C)")
plt.show()

# Rachel, want to add your code here?

##############################################################################
# NOTE: ZIP model begins here!
##############################################################################

# do zero-inflated Poisson (ZIP) model (rate model with chlorophytes as rate):

# Run ZIP model
zip_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=X_train,
                                              exog_infl=X_train,
                                              inflation='logit').fit()
print(zip_training_results.summary())

# calculuate root mean squared prediction error
zip_predictions = zip_training_results.predict(X_test, exog_infl=X_test)
predicted_counts = np.round(zip_predictions)
actual_counts = y_test["y_te"]
print('ZIP RMSE='+str(np.sqrt(np.sum(np.power(np.subtract(
                                              predicted_counts,
                                              actual_counts), 2)))))

# prediction plot 1
fig = plt.figure()
fig.suptitle('Predicted vs. Actual Chlorophyte Concentrations')
predicted, = plt.plot(X_test.index, predicted_counts, 'go-',
                      label='Predicted')
actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual')
plt.xlabel("Indices")
plt.ylabel("Chlorophyte Concentration ($\mu$g/L)")
plt.legend(handles=[predicted, actual])
plt.show()

# prediction plot 2 (bubble plot)
diff = np.absolute(actual_counts - predicted_counts)
test_data = {'Temperature ($^\circ$C)': X_test["x2_te"],
             'Salinity (% NaCl)': X_test["x1_te"], 'Difference': diff}
test_data_df = pd.DataFrame(data=test_data)
sns.scatterplot(data=test_data_df, x="Temperature ($^\circ$C)",
                y="Salinity (% NaCl)", size="Difference", legend=False,
                sizes=(20, 2000))
plt.title("Predicted vs. Actual Chlorophyte Concentration")
plt.show()

# manually run predictions for particular values
# note that param's 3 through 6 indices are intercept, salinity, temp

# pred_chloro = np.exp(np.matmul([1, 23.5, 28.5],
#                      zip_training_results.params[3:6]))
# print("ZIP prediction")
# print(pred_chloro)

##############################################################################
# TODO LIST:

# - TODO: we get same ZIP results each time for same data; how much should ZIP
#         results be changing for diff data?? Currently have it set to same
#         data each time - need to bootstrap
# - TODO: deal with negative fitted values
# - TODO: run more diagnostic plots and test goodness of fit (esp for ZIP)
# - TODO: calculate residuals? more error calculations?
# - TODO: double check sample size (ok to have only 36 records?)
# - TODO: improve predictions and MSPE
# - TODO: get predict() to work for ZIP
# - TODO: see other lists
# - TODO: decide if training set should be basis of model if we publish

##############################################################################
