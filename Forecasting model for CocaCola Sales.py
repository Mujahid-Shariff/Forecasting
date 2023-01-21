# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:59:42 2022

@author: Mujahid Shariff
"""

import pandas as pd
CocaCola = pd.read_csv("CocaCola_Sales_Rawdata.csv")
CocaCola.shape
list(CocaCola)
CocaCola.Sales.plot()

CocaCola['Quarter'].value_counts()

import matplotlib.pyplot as plt

#Describing the data
CocaCola.describe()

#Here in our original dataset we have added each column for each month, T and TSquare values for each passengers data

#%B : Returns the full name of the month, e.g. September. %w : Returns the weekday as a number, from 0 to 6, with Sunday being 0. 
#%m : Returns the month as a number, from 01 to 12. 
#%p : Returns AM/PM for time.
#%y : Returns the year in two-digit format, that is, without the century.

#EDA
#Line plot
CocaCola.Sales.plot()

#Kernel density estimate (KDE) plot 
CocaCola.plot(kind="kde")

#Histogram
CocaCola.Sales.hist()

#Heatmap for our data to understand maximum numbers of passengers for every month/year
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_y_quarter = pd.pivot_table(data=CocaCola,values="Sales",index="Quat", columns="Year",fill_value=0)
sns.heatmap(heatmap_y_quarter,annot=True,fmt="g") #fmt is format of the grid values

# Boxplot for every month, year vs Passengers
plt.figure(figsize=(8,6))
#plt.subplot(211)
sns.boxplot(x="Quarter",y="Sales",data=CocaCola)
#plt.subplot(212)
plt.figure(figsize=(12,6))
sns.boxplot(x="Quarter",y="Sales",data=CocaCola)

#Line plot
plt.figure(figsize=(12,3))
sns.lineplot(x="Quarter",y="Sales",data=CocaCola)
CocaCola

#==============================================================================
# Splitting data
CocaCola.shape
Train = CocaCola.head(36) #taking 84 since data contains 96 columns, we are predicting for next 12 months, 96-12=84
Test = CocaCola.tail(6)
Test


#Inserting the stats model for our dataset
import statsmodels.formula.api as smf 

#Linear Model
import numpy as np

#lm.fit(x,y) #this is how we fit in Linear model
#y~x #this is how we fit in stats model

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
#np.sqrt(np.mean((Test['Footfalls']-np.array(pred_linear))**2))
rmse_linear

#Exponential Model
Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#Quadratic Model
Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
#pred_Quad = pd.Series(Exp.predict(pd.DataFrame(Test[["t","t_square"]))) # we hve to verify
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

#Additive seasonality Model
add_sea = smf.ols('Sales~+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

#Additive Seasonality Quadratic Model
add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative Seasonality Model
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


#Multiplicative Additive Seasonality Model
Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

#Compare the results with each models RMSE to view consolidated file
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data) #to understand the type of data, here its Dict

table_rmse=pd.DataFrame(data) #converting to DataFrame
table_rmse.sort_values(['RMSE_Values']) #sort by RMSE low to high

#               MODEL  RMSE_Values
#4  rmse_add_sea_quad   283.061961 <==== Lowest RMSE
#2          rmse_Quad   485.140670
#0        rmse_linear   667.425698
#3       rmse_add_sea  1895.559313
#6  rmse_Mult_add_sea  4565.576579
#1           rmse_Exp  4565.587722
#5      rmse_Mult_sea  4572.794690

#### Predict for new time period, new prediction for next Quarter3, Q4 - 1996, Q1-Q4 - 1997
#we have added a new predicted file, with next year data
new_data = pd.read_csv("Predict_New_Coca.csv")
new_data

#Building the model on entire data set
model_full = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=CocaCola).fit()

pred_new  = pd.Series(model_full.predict(new_data))
pred_new #This contains predictions for next 12 months

#===========================================

#Adding forecasted values to our dataset pred_new
new_data["forecasted_Sales"] = pd.Series(pred_new)

#concat the predicted with original
new_var = pd.concat([CocaCola,new_data])
new_var.shape
new_var.head()
new_var.tail()

#Plot for predictions for the next 12 months, Jan 2003 to Dec 2003
new_var[['Sales','forecasted_Sales']].reset_index(drop=True).plot()

#Exporting the csv to view the data 
new_var.to_csv("E:\\DS - ExcelR\\Assignments\\Forecasting\\New data with predictions.csv")
               
#######################################################################################


