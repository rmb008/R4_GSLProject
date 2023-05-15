# Modeling the microbiome of Utah’s Great Salt Lake: A regression analysis of key abiotic factors impacting growth of Dunaliella green algae in the GSL’s South Arm

## Description
Since 1987, water levels in the Great Salt Lake have dropped by 20 feet, presenting many serious ecological, public health, and economic concerns. Here, we focus on the impact of dropping water levels on
the microbe Dunaliella viridis, an important primary producer in the South Arm of the Great Salt Lake. As water levels drop, water temperatures and salinity increase, and we aim to predict how Dunaliella
populations will change with increasing temperature and salinity. In our project, we develop 2 multiple regression models to assess the impact of water temperature and salinity on Dunaliella concentration
using data from [Belovsky et al.'s 2011 paper](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/ES10-00091.1) in the journal Ecosphere. Specifically, we developed a multiple linear regression model and a zero-inflated Poisson regression model.

## Required Packages:
The following packages need to be installed to run this script, if they are not installed already:
- matplotlib
- pandas
- numpy
- scipy
- sklearn
- seaborn
- mpl_toolkits.mplot3d
- patsy
- statsmodels
- math
- random

Installation can be completed using ``conda install`` or ``pip install``. The environment for this script (GSL-env) is stored in environment.yml.

## Main Script
The main script to run in this repository is ``regression.py``. This script takes in the data from the Belovsky paper, which is stored in the csv file named ``field_data.csv`` and is included in the repository
for convenience. ``regression.py`` performs our entire data analysis, first cleaning and processing the data in preparation for regression, then performing exploratory data analysis through the generation of
histograms, and finally implements and produces results and plots for both the multiple linear regression and zero-inflated Poisson regression models. Expected output includes:
- 3 histograms for the distributions of salinity, temperature, and chlorophyte concentration
- 1 histogram that overlays the previous 3 histograms 
- 1 histogram that produces subplots of the 3 hisotgrams of each variable
- p-values, regression intercept and coefficients, and $R^2$ value for multiple linear regression
- 2 plots graphing actual chlorophyte concentration vs temperature and salinity, respectively
- 2 plots graphing fitted chlorophyte concentration vs temperature and salinity, respectively
- 1 plot graphing actual vs fitted chlorophyte concentrations
- 1 plot graphing temperature vs salinity
- 1 plot that produces a 3D rendering of the multiple linear regression model (that you can drag and rotate)
- 1 plot that produces 3 different angles of the 3D rendering
- summary of the zero-inflated Poisson regression (including pseudo $R^2$, p-values, regression coefficients/intercept)
- 1 plot graphing both actual and fitted chlorophyte concentrations on the y axis and index of the dataframe on the x-axis
- 1 plot graphing salinity vs temperature, with bubble size corresponding to absolute prediction error between predicted and actual chlorophyte concentrations for the given (temperature, salinity) pairs

There is one more useful point regarding the ``regression.py`` script. Specifically, the multiple linear regression model uses both the full dataset and the training/testing datasets, depending on the context. 
If the goal is to calculate prediction error or generate predictions, the user should comment out Lines 161 and 163 and uncomment Lines 160 and 162 to run the linear regression model using the training set.
Then the user can uncomment Lines 168-172 to make a prediction for what happens to chlorophyte concentration when temperature reaches 28.5 degrees Celsius and when salinity reaches 23.5%. Note that any temperature and salinity values can
be used here, but Dunaliella do not survive beyond 40 degrees Celsius and 30% salinity, so increasing beyond these values is not meaningful. Furthermore, the user can uncomment Lines 201-208 to calculate the 
root mean squared prediction error for the linear model. Note that all of the output plots and results (except predictions and prediction error) for the linear model are generated using the whole dataset
rather than the training set. In contrast, the zero-inflated Poisson model is trained directly on the training set. If the user uncomments Lines 413-416, they will produce the prediction from the Poisson model
for chlorophyte concentration for when temperature is 28.5 degrees Celsius and salinity is 30%.

## Contact Information
Questions about our code? Contact Vanessa Maybruck (vanessa.maybruck@colorado.edu) or Rachel Billings (rachel.billings@colorado.edu).
