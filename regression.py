# Reads in the field data from the Belovsky Ecosphere paper and runs a
# multiple linear regression and zero inflated Poisson regression with temp,
# salinity as predictors and micrograms of Chlorophytes per L as the response.

# Import packages
import matplotlib.pyplot as plt
from matplotlib import font_manager
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

# produce histograms for chlorophyte concentration, temp, and salinity
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')

# Data to pandas dataframe
xdf = pd.DataFrame(X)
ydf = pd.DataFrame(Y)
xydf = pd.concat([xdf, ydf], axis=1)

# Salinity histogram
fig = plt.figure(figsize=(5, 4))
sns.histplot(data = xydf,
             x = "SALINITY (%)",
             kde = True,
             color = 'b')
plt.title("")
fig.suptitle('Salinity', fontsize = 22)
fig.tight_layout()
plt.show()

# Temperature histogram
fig = plt.figure(figsize=(5, 4))
sns.histplot(data = xydf,
             x = "TEMPERATURE (C) ",
             kde = True,
             color = 'r')
plt.title("")
fig.suptitle('Temperature', fontsize = 22)
fig.tight_layout()
plt.show()

# Chlorophyte concentration histogram
fig = plt.figure(figsize=(5, 4))
sns.histplot(data = xydf,
             x = "CHLOROPHYTES (ug/l)",
             kde = True,
             color = 'g')
plt.title("")
fig.suptitle('Chlorophyta', fontsize = 22)
fig.tight_layout()
plt.show()

# Overlay with all histograms together
fig = plt.figure(figsize=(10, 4))
sns.histplot(data = xydf, kde = True)
fig.suptitle('Salinity, Temperature, and Chlorophyta Histograms', fontsize = 22)
plt.title('Overlay', fontsize = 18)
fig.tight_layout()
plt.show()

# Subplots of all histograms
fig, axes = plt.subplots(1, 3, figsize = (10,6))
fig.suptitle('Salinity, Temperature, and Chlorophyta Histograms', fontsize = 22)
sns.histplot(data = xydf,
             x = "SALINITY (%)",
             kde = True,
             ax = axes[0],
             color = 'b')
sns.histplot(data = xydf,
             x = "TEMPERATURE (C) ",
             kde = True,
             ax = axes[1],
             color = 'r')
sns.histplot(data = xydf,
             x = "CHLOROPHYTES (ug/l)",
             kde = True,
             ax = axes[2],
             color = 'g')
axes[0].set_title('Salinity', fontsize = 18)
axes[1].set_title('Temperature', fontsize = 18)
axes[2].set_title('Chlorophyta', fontsize = 18)
plt.show()

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

# plot temp vs salinity (linear fit does not work here)
m, n = np.polyfit(X["SALINITY (%)"], X["TEMPERATURE (C) "], 1)
plt.scatter(X["SALINITY (%)"], X["TEMPERATURE (C) "])
plt.plot(X["SALINITY (%)"], m*X["SALINITY (%)"]+n)
plt.title("Temperature vs. Salinity")
plt.xlabel("Salinity (% NaCl)")
plt.ylabel("Temperature ($^\circ$C)")
plt.show()

# prepare for 3D plotting of MLR model
# Set Seaborn theme
sns.set(style = "whitegrid")
sns.set_theme(context='notebook', style='darkgrid', palette='viridis',
              font_scale=1, color_codes=True, rc=None)
# Font setting
flist = font_manager.get_font_names()
# print(flist)
font = font_manager.FontProperties(family = 'Trebuchet MS')
file = font_manager.findfont(font)
prop = font_manager.FontProperties(fname = file)
sns.set(font = prop.get_name())

# function to make MLR 3D plot - single plot, this is the plot that you
# can rotate
def mlrplot_single(x, y, z, xx_pred, yy_pred, fit_data):
    fig = plt.figure(figsize = (12, 10))
    seaborn_plot = plt.axes(projection='3d')
    print(type(seaborn_plot))
    seaborn_plot.scatter3D(x, y, z, s = 150, c = 'teal', depthshade = True,
                           edgecolor = 'black')

    seaborn_plot.scatter(xx_pred.flatten(),
                yy_pred.flatten(),
                fit_data,
                facecolor = 'grey',
                s = 15,
                edgecolor='white',
                alpha = .25)
    seaborn_plot.set_xlabel('Salinity (% NaCl)', fontsize=18, labelpad=15)
    seaborn_plot.set_ylabel('Temperature ($^\circ$C)', fontsize=18,
                            labelpad=15)
    seaborn_plot.set_zlabel('Chlorophytes ($\mu$g/L)', fontsize=18,
                            labelpad=15)
    seaborn_plot.locator_params(nbins = 9, axis='x')
    seaborn_plot.locator_params(nbins = 9, axis='y')
    seaborn_plot.locator_params(nbins = 9, axis='z')
    fig.suptitle('Multiple Linear Regression 3D Plot \n $R^2 = %.2f$' % r2,
                 fontsize=22)
    fig.tight_layout()
 #   fig.update_layout(margin=dict(l=5, r=5, b=5, t=5))
    fig.show()

    return

# function to make MLR 3D plot - multiple plots (view 3D plot from multiple
# angles)
def mlrplot(x, y, z, xx_pred, yy_pred, fit_data):
    print(plt.style.available)
    plt.style.use('default')

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.plot(x, y, z,
                color='red',
                zorder=15,
                linestyle='none',
                marker='o',
                alpha=0.8)
        ax.scatter(xx_pred.flatten(),
                   yy_pred.flatten(),
                   fit_data,
                   facecolor = 'grey',
                   s = 10,
                   edgecolor='white',
                   alpha = .3)
        ax.set_xlabel('Salinity (% NaCl)', fontsize=12)
        ax.set_ylabel('Temperature ($^\circ$C)', fontsize=12)
        ax.set_zlabel('Chlorophytes ($\mu$g/L)', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')
        #ax.invert_xaxis()
        #ax.invert_yaxis()
       # ax.invert_zaxis()
       # plt.ylim(max(y), min(y))
    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=0, azim=90)
    ax3.view_init(elev=60, azim=165)
    fig.suptitle('$R^2 = %.2f$' % r2, fontsize=22)
    fig.tight_layout()
    fig.show()
    return

# re-run MLR to generate plots
x = X["SALINITY (%)"]
y = X["TEMPERATURE (C) "]
z = Y
# salinity range
x_pred = np.linspace(np.min(x), np.max(x), np.size(x))
# temperature range
y_pred = np.linspace(np.min(y), np.max(y), np.size(y))

xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

regr = linear_model.LinearRegression()
model = regr.fit(X, Y)
fitted2 = model.predict(model_viz)
r2 = model.score(X, Y)

# run function to make single (rotating) mlr plot
mlrplot_single(x, y, z, xx_pred, yy_pred, fit_data = fitted2)
# run function to make multiple mlr plots
mlrplot(x, y, z, xx_pred, yy_pred, fit_data = fitted2)

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

# prediction plot 1 (with indices on x axis)
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
