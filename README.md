# Spring 2022: Case Shiller Housing Prediction
Matt Wolff, Laywood Fayne and Ari Webb

## Introduction
We used various monthly municipal and macro-economic metrics to predict housing
market performance in American cities by the Case Shiller Index. The Case-Shiller index uses repeat sales of single-family homes to track trends in home
pricing. We started by collecting city level data including unemployment data, consumer price index data (CPI), crime
rates and patent data. We also used data that plots more general economic trends, such as US stock
market data. 

## Methods

We used three different models on the data. Linear regression, a neural net and Vector Autoregression. I implemented the VAR. VAR is a time series prediction algorithm that models the future values of features as linear combinations of the previous values of that feature and past values of other features. The driver code for this model including various preprecessing methods including Grangers Causation analysis is found in data/final_var.py

## Results

The model forecasts can be found in /data/forecasts. Forecasts include plots for every feature in each city going forward a year.
