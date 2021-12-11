# rateExtrapolation.py
[![Build Status](https://app.travis-ci.com/sarahwaity/rateExtrapolation.svg?branch=main)](https://app.travis-ci.com/sarahwaity/rateExtrapolation)
[![license](https://img.shields.io/github/license/sarahwaity/rateExtrapolation.svg?style=flat-square)](https://github.com/sarahwaity/rateExtrapolation.svg/main/LICENSE)

## Video Example
https://youtu.be/ykR0b9ED8NI

## Colab Notebook


## Executive Summary
The purpose of this package is to enable users to very simply determine reaction kinetics from experimental data. When the user inputs the required inputs, they will be returned an estimation of the rate constants that underly their observation. 




## User Profile
### Experimentalist
The expected experimentalist user will have some coding experience in python and familiarity with the tellurium package. Ideally, the user will be able to import python packages and understand how to call python functions. The user should be able to write tellurium antimony and identify rates as strings. However, the function is meant to be simple enough to be a “plug and chug” analysis of rates. 


## Use Cases
### Rate Constant Determination
#### Objective
The objective of this function is to provide approximate rate constants that are governing the changes affecting their observed substrates. The function will determine rate constants that describe the behavior of their systems and provide statistical confidences of model/parameter fitness.  
#### Expected Interactions
The users will input experiment derived data as a single list describing the observed concentrations of a substrate. Users will call the function and input experimental data as data type pd.DataFrame OR csv. They will also supplement an antimony string with unknown constants and a list of parameters (constants) to be estimated as a string. Output will include plots of experimental data vs. the model with the best estimated rates. It also generates a histogram of extrapolated rate values with heatmapped R^2. Counts will be the number of folds.  Table generated will pull data from SBstoat model fitness information and provide rates determined, fitting method, R2,  Chi-squared, Reduced chi-square, Akaike info crit, Bayesian info crit. The user will need to be able to understand the plots and make educated decisions of the rates from the statistics and modeling. 
