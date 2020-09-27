# Introduction
Project: French Motor Claims  
Author: Bhishan Poudel, Ph.D Physics  
Goal: Implement Frequency model, Severity model and Pure Premium Model  
Tools: pandas, scikit-learn, xgboost  
References:
- https://www.kaggle.com/floser/french-motor-claims-datasets-fremtpl2freq
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html

|  Notebook | Rendered   | Description  |  Author |
|---|---|---|---|
| a01_data_cleaning.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/a01_data_cleaning.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/a01_data_cleaning.ipynb)  | ohe, kbin, logscaling  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b01_freq_modelling.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b01_freq_modelling.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b01_freq_modelling.ipynb)  | Poisson  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b02_severity_modelling.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b02_severity_modelling.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b02_severity_modelling.ipynb)  | Gamma  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b03_pure_premium_modelling.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b03_pure_premium_modelling.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b03_pure_premium_modelling.ipynb)  | Poisson*Gamma and Tweedie  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b04_tweedie_vs_freqSev.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b04_tweedie_vs_freqSev.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b04_tweedie_vs_freqSev.ipynb)  | comparison   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| b05_lorentz_curves_comparison.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b05_lorentz_curves_comparison.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/b05_lorentz_curves_comparison.ipynb)  | Lorentz Curve  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| c01_tweedie_xgboost.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/c01_tweedie_xgboost.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/c01_tweedie_xgboost.ipynb)  | 'objective':'reg:tweedie'  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |


# Data
- [openml french motor freq](https://www.openml.org/d/41214) (Multiple features.)
- [openml french motor severity](https://www.openml.org/d/41215) (Two features Id policy and Claim Amount.)

# Data Cleaning
Some of the features are chosen for modelling.
```
one hot encoding = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
kbins discretizer = ["VehAge", "DrivAge"]
log and scaling = ["Density"]
pass through =  ["BonusMalus"]
```

# Frequency Modelling (Poisson Distribution)
For the frequency modelling I used Poisson distribution.

| Metric | train | test |
| :---|:---|:---|
| D2 | 0.051384 | 0.048138 |
| mean_absolute_error | 0.232085 | 0.224547 |
| mean_squared_error | 4.738399 | 2.407906 |

# Severity Modelling (Gamma Distribution)
I used Gamma Regressor for the frequency modelling.

| Metric | train | test |
| :---|:---|:---|
| D2 | 3.638157e-03 | -4.747382e-04 |
| mean_absolute_error | 1.859814e+03 | 1.856312e+03 |
| mean_squared_error | 4.959565e+06 | 4.827662e+06 |

Here, the D-squared value for test is too worse than the training data. This is because while fitting the train data we have used only claims with `ClaimAmount > 0`. This model is calculating `average claim amount per claim only when claim is more than zero`. This model CAN NOT predict average claim per policy in general.

# Xgboost Tweedie Regression
I used xgboost with `objective='reg:tweedie'` and applied following offest
```python
dtrain.set_base_margin(np.log(df_train['Exposure'].to_numpy()))
dtest.set_base_margin(np.log(df_test['Exposure'].to_numpy()))
```
to model the pure premium. Xgboost does not implement the D-squared value. The model performance is given below:
| Metric | train | test |
| :---|:---|:---|
| mean_absolute_error | 1.760538e+03 | 1.588351e+03 |
| mean_squared_error | 1.481952e+09 | 1.659363e+08 |
