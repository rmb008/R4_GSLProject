# This file reads in the field data from the Belovsky Ecosphere paper and runs a multiple linear
# regression with temp, salinity as predictors and micrograms of Chlorophytes per L as the response.

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import linear_model

lmdata = pd.read_csv("field_data.csv")
#print(lmdata)

# replace first header (need second header to pull our needed columns
new_header = lmdata.iloc[0]
lmdata = lmdata[1:]
lmdata.columns = new_header
#print(lmdata)

# remove last row (it's the sum of all previous rows for each column)
lmdata = lmdata.drop(labels=157, axis=0)

# select the columns we need (note: the csv file has an extra space at the end of the temp string
lmdata_reduced = lmdata[["SALINITY (%)", "TEMPERATURE (C) ", "CHLOROPHYTES (ug/l)"]]
#print(lmdata_reduced)

# remove missing values (all denoted by 99999)
lmdata_filter1 = lmdata_reduced[lmdata_reduced["SALINITY (%)"] != '"99999"']
#print(lmdata_filter1)
lmdata_filter2 = lmdata_filter1[lmdata_filter1["TEMPERATURE (C) "] != '"99999"']
lmdata_filter3 = lmdata_filter2[lmdata_filter2["CHLOROPHYTES (ug/l)"] != '"99999"']
# print(lmdata_filter3)
# print(lmdata_filter3.shape) # 58 observations of 3 variables is sufficient for regression

# need to re-index this for regression
lmdata = lmdata_filter3.reset_index()
#print(lmdata)
# drop extra index column
lmdata = lmdata.drop(labels="index", axis=1)
#print(lmdata)

# assign predictor (X) and response (Y) variables for regression - note that X is an array
X = lmdata[["SALINITY (%)", "TEMPERATURE (C) "]]
Y = lmdata["CHLOROPHYTES (ug/l)"]

# run the regression using sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

# print parameters/coefficients
print(regr.coef_)

# TODO: need to split data first and figure out how to plot MLR - want to get a plot and estimate of its goodness before tomorrow

# TODO: check where the values are being pulled from (add month and year var's back in to make
# sure they are not all from the same season, may want to check if they say if all samples
# taken from same place and depth too)

# TODO: don't forget to split into testing and training sets before predicting, calculuate MSPE, etc
# TODO: merge environments or rename or something (currently, this works with sklearn env but needs to be a unique env)-
# currently using sklearn env (I think that is the right one, and it is connected to the environment.yml file)
