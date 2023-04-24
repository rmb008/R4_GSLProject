# This file reads in the field data from the Belovsky Ecosphere paper and runs a multiple linear
# regression with temp, salinity as predictors and micrograms of Chlorophytes per L as the response.

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd
from sklearn import linear_model
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

lmdata = pd.read_csv("field_data.csv")
#print(lmdata)

# replace first header (need second header to pull our needed columns)
new_header = lmdata.iloc[0]
lmdata = lmdata[1:]
lmdata.columns = new_header
#print(lmdata)

# remove last row (it's the sum of all previous rows for each column)
lmdata = lmdata.drop(labels=157, axis=0)

# select the columns we need (note: the csv file has an extra space at the end of the temp string)
lmdata_reduced = lmdata[["MONTH", "YEAR", "SALINITY (%)", "TEMPERATURE (C) ", "CHLOROPHYTES (ug/l)"]]
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

# drop extra index column from re-indexing
lmdata = lmdata.drop(labels="index", axis=1)
#print(lmdata)

# convert data from string to numeric
lmdata["SALINITY (%)"] = lmdata["SALINITY (%)"].astype(float)
lmdata["TEMPERATURE (C) "] = lmdata["TEMPERATURE (C) "].astype(float)
lmdata["CHLOROPHYTES (ug/l)"] = lmdata["CHLOROPHYTES (ug/l)"].astype(float)
#print(type(lmdata["SALINITY (%)"][1]))

#print(lmdata)

# try dropping entries with zero chlorophytes:
lmdata = lmdata[~(lmdata['CHLOROPHYTES (ug/l)'] == 0)]
#print(lmdata)

# need to re-index this for regression
lmdata = lmdata.reset_index()
#print(lmdata)

# drop extra index column from re-indexing
lmdata = lmdata.drop(labels="index", axis=1)
print(lmdata)

# assign predictor (X) and response (Y) variables for regression - note that X is an array
X = lmdata[["SALINITY (%)", "TEMPERATURE (C) "]]
Y = lmdata["CHLOROPHYTES (ug/l)"]

# log transform the chlorophytes variable since it has a lot of small values near 0
Y = np.log(Y) # note that this is the natural log

# code needed for 3D plot #TODO: adjust to our data
x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(6, 24, 30)   # range of porosity values
y_pred = np.linspace(0, 100, 30)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

# run the regression using sklearn
regr = linear_model.LinearRegression()
model = regr.fit(X, Y)
fitted = model.predict(X) # these are the chlorophyte values predicted from/fitted to the regression model
#print(fitted) #TODO: deal with negative fitted values (the ones that are negative without the log transform)

# print parameters/coefficients
#print(regr.coef_)

# print intercept of fit line
#print(regr.intercept_)

# print R^2
#print(model.score(X, Y)) # R^2 is currently low, possibly poor model fit

# plot X vs Y (these are just values from the actual data)
#plt.scatter(X["SALINITY (%)"], Y) # salinity and chlorophytes
#plt.show()
#plt.scatter(X["TEMPERATURE (C) "], Y) # temp and chlorophytes
#plt.show()

# plot X vs fitted
plt.scatter(X["SALINITY (%)"], fitted)
plt.show()
plt.scatter(X["TEMPERATURE (C) "], fitted)
plt.show()

# plot actual vs fitted #TODO: fit a nonparametric line to this
plt.scatter(Y, fitted)
plt.show()

# plot temp vs salinity (proxy test for correlation, official correlation test coming) - almost linear: PROBLEM but only for inference (not prediction)
#plt.scatter(X["SALINITY (%)"], X["TEMPERATURE (C) "])
#plt.show()

# MLR 3D plot
plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(X["SALINITY (%)"], X["TEMPERATURE (C) "], Y, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(X, Y, fitted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Salinity (% NaCl)', fontsize=12)
    ax.set_ylabel('Temperature (deg. C)', fontsize=12)
    ax.set_zlabel('Chlorophytes (ug/L)', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)
ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax2.transAxes, color='grey', alpha=0.5)
ax3.text2D(0.85, 0.85, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax3.transAxes, color='grey', alpha=0.5)

ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()

# TODO: check sample size (ok to have only 36 records?)

# TODO: run diagnostic plots

# TODO: 3D plots of MLR

# TODO: don't forget to split into testing and training sets before predicting, calculuate MSPE, etc
