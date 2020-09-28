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
| c01_xgboost_tweedie.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/c01_xgboost_tweedie.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/c01_xgboost_tweedie.ipynb)  | 'objective':'reg:tweedie'   | [Bhishan Poudel](https://bhishanpdl.github.io/)  |
| d01_gam_grid_search.ipynb  | [ipynb](https://github.com/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/d01_gam_grid_search.ipynb), [rendered](https://nbviewer.jupyter.org/github/bhishanpdl/Project_French_Motor_Claims/blob/master/notebooks/d01_gam_grid_search.ipynb)  | n_splies=10, grid_search  | [Bhishan Poudel](https://bhishanpdl.github.io/)  |


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

# Results
|Module | Distribution | y_train | sample_weight | train D2 | test D2 | train MAE | test MSE | train MAE | test MSE |
| :---|:---|:---|:---|:---|:---| :---|:---| :---|:---|
|sklearn | Frequency Modelling (Poisson Distribution) | df_train['Frequency']  | df_train['Exposure']|0.051384 | 0.048138 | 0.232085 | 0.224547  | 4.738399 | 2.407906 |
|sklearn | Severity Modelling (Gamma Distribution) | df_train.loc[mask_train, 'AvgClaimAmount'] | df_train.loc[mask_train, 'ClaimNb'] | - | 3.638157e-03 | -4.747382e-04 | 1.859814e+03 | 1.856312e+03 | 4.959565e+06 | 4.827662e+06 |
|sklearn|Pure Premium Modelling (TweedieRegressor) | df_train['PurePremium'] | df_train['Exposure'] | 2.018645e-02 | 1.353285e-02 | 6.580440e+02 | 4.927505e+02 | 1.478259e+09 | 1.622053e+08 |
|xgboost | Xgboost Tweedie Regression | dtrain.set_base_margin(np.log(df_train['Exposure'].to_numpy())| dtest.set_base_margin(np.log(df_test['Exposure'].to_numpy())) | - | - | 1.760538e+03 | 1.588351e+03 | 1.481952e+09 | 1.659363e+08 |
|pygam | GAM Linear Model | - | - | - | - | 1.686438e+02 | 1.655408e+02 | 1.785332e+06 | 1.647533e+06 |




















