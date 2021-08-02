import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Reading the .csv files into pandas dataframe

pc = pd.read_csv("PowerCurve.csv")
se = pd.read_csv("SiteEnergy.csv")
sd = pd.read_csv("SiteData.csv")
wd = pd.read_csv("WindData.csv")

# Transforming hourly wind speed using the power curve to get hourly energy production
X = pc.iloc[:, 0:1].values
y = pc.iloc[:, 1].values


# Considering the curve between the cut-in speed and the cut-off speed

zero_index = len(y)-1
for i in range(len(y)-1,0,-1):
    if y[i] != 0:
        zero_index = i
        break
    
y = y[:zero_index+1]
X = X[:zero_index+1]

zero_index = 0
for i in range(0,len(y)):
    if y[i] != 0:
        zero_index = i
        break

y = y[zero_index-1:]
X = X[zero_index-1:]

# Implementation of Polynomial Regression on the power curve to train the model

poly = PolynomialFeatures(degree = 8) 
X_poly = poly.fit_transform(X) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 

# Plotting the polynomial curve

plt.scatter(X, y, color = 'blue')
plt.xlabel('Wind Speed(m/s)')
plt.ylabel('Power(kW)') 
plt.suptitle(' GE 1.5SLE Powercurve')
plt.show()

# Running the wind data through the polynomial regression model
# Since the height of the turbine is given as 80m, We will just run the model on the wind-speed at 50m

pw = wd.iloc[:,5].values
pw  = pw.reshape(-1,1)
new_x = poly.fit_transform(pw) 
output = lin2.predict(new_x)

# Any wind speed values outside the cut-off and cut-in are set to zero    

wd['prediction'] = output
wd.loc[wd.NW_spd_50m<3 , 'prediction'] = 0
wd.loc[wd.NW_spd_50m>30 , 'prediction'] = 0

# Aggregate hourly energy production to the monthly level

wd[['yyyy', 'mm', 'na']] = wd['Time'].str.split('-', expand=True)
wd = wd.groupby(by = ['yyyy','mm'], as_index = False)['prediction'].sum()

# Converting the kWh to MWh
# Given number of turbines as 134, multiplying 134 to the predicted energy     
wd['prediction'] = wd['prediction']*0.134 # 134/1000

# Using R-squared value to measure how close the predicted data is to the observed data

test  = wd.iloc[345:441, 2]
obs = se.iloc[9:105, 1]
r = r2_score(obs, test)
print (r) # r = 0.6597780067311735

#Compare modeled monthly energy to the publicly reported observed energy

site_ene = se
pred_ene = wd.iloc[336:441,:]
d1 = pred_ene.iloc[: ,2].tolist()
site_ene[['yyyy', 'mm']] = se['Month'].str.split('-', expand=True) 
d = se.iloc[: ,1]
year = site_ene.iloc[:,3]
month = site_ene.iloc[:,4]
x = [0] * len(year)
for i in range(0,len(year)):
    x[i] = float(year[i]) + float(month[i])/12.0

# Visualization of modeled monthly energy to the publicly reported observed energy
    
visualization = pd.DataFrame({'Year':x,'Energy.MWh_obs':d,'Energy.MWh_pred':d1})
visualization.plot(x = 'Year',y = ['Energy.MWh_obs','Energy.MWh_pred'], title = 'Logan Wind Energy Site Production')

