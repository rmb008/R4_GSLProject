# Modeling the microbiome of Utah’s Great Salt Lake: A regression analysis of key abiotic factors impacting growth of Dunaliella green algae in the GSL’s South Arm

## Description
Since 1987, water levels in the Great Salt Lake have dropped by 20 feet, presenting many serious ecological, public health, and economic concerns. Here, we focus on the impact of dropping water levels on
the microbe Dunaliella viridis, an important primary producer in the South Arm of the Great Salt Lake. As water levels drop, water temperatures and salinity increase, and we aim to predict how Dunaliella
populations will change with increasing temperature and salinity. In our project, we develop 2 multiple regression models to assess the impact of water temperature and salinity on Dunaliella concentration
using data from Belovsky et al.'s 2011 paper in the journal Ecosphere. Specifically, we developed a multiple linear regression model and a zero-inflated Poisson regression model.

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
