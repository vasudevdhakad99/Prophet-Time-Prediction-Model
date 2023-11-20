
import pandas as pd
import numpy as np
import warnings
import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib

from prophet import Prophet

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


from statsmodels.tsa.arima.model import ARIMA
df_=pd.read_excel(r'D:\Users\star26\Documents\SALES_ML_MODEL.xlsx')
#df.info()
#df.describe()
df_.columns


df_.describe()

# Step 3: Calculate the mean and standard deviation of the "Qty" column
qty_mean = df_['Qty'].mean()
qty_std = df_['Qty'].std()

# Step 4: Define a threshold to identify outliers (e.g., 3 times the standard deviation)
threshold = 3

# Step 5: Filter the DataFrame to keep only the rows where "Qty" values are within the threshold
df = df_[abs(df_['Qty'] - qty_mean) <= threshold * qty_std]


drop_columns = ['Month', 'Year']
df_.columns
df.drop(drop_columns,axis=1,inplace =True)

import re

pattern = r'2021|2022|2023'

df5 = df[df['Invoice Date'].dt.year.astype(str).str.contains(pattern, regex=True)]
df5['Indent Date']=df5['Invoice Date']


############################################################################
pd.to_datetime(df5['Indent Date'])
#df['Indent Date'] = df.sort_values('Indent Date')
############################################################################
df1=df5

df1.describe()

df1.columns


# Add new columns for month, year, and week
df1['Month'] = df1['Indent Date'].dt.month
df1['Year'] = df1['Indent Date'].dt.year
df1['Week'] = df1['Indent Date'].dt.strftime('%U')

# Optionally, if you want to format the week number as an integer:
df1['Week'] = df1['Indent Date'].dt.isocalendar().week

# Optionally, if you want to format the week number with leading zeros (e.g., 'Week 01'):
df1['Week'] = df1['Indent Date'].dt.strftime('Week %U')

# Optionally, if you want to combine month and year in a single column:
df1['Month_Year'] = df1['Indent Date'].dt.strftime('%B %Y')

# Optionally, if you want to combine month, year, and week in a single column:
df1['Month_Year_Week'] = df1['Indent Date'].dt.strftime(' %Y-Week %U')

# Display the DataFrame with the new columns
print(df1)

df1.drop(['Indent Date','Item Name','Party Name','Year','Month','Week','Month_Year'],axis=1,inplace=True)

df1['Qty'] = df1['Qty']/1000

df1 = df1.groupby('Month_Year_Week')['Qty'].sum()

#df1.set_index('Indent Date')
df2 = df1.to_frame()

df2

df2.plot()                                              
#df2['Sales'] = df2['Qty']




# Create and fit the Prophet model
model = Prophet()
df2.reset_index(drop=False, inplace=True)

df2.rename(columns={'Month_Year_Week':'ds','Qty':'y'},inplace=True)

import pandas as pd

def convert_to_datetime(date_str):
    year_str, week_str = date_str.split('-Week ')
    year = int(year_str)
    week = int(week_str)
    return pd.to_datetime(f"{year}-W{week:02d}-1", format='%Y-W%W-%w')

df2['ds'] = df2['ds'].apply(convert_to_datetime)

#df2['ds'] = pd.to_datetime(df2['ds'])
#df2.reset_index(drop=True, inplace=True)
df2.info()

model.fit(df2)

# Generate future dates for prediction
future = model.make_future_dataframe(periods=24, freq='W')

# Make predictions
forecast = model.predict(future)

forecast.head()

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forecast[['ds','yhat','yhat_lower','yhat_upper']].head()

model.plot(forecast)





# Extract the actual values and forecasted values
actual_values = df2['y'].values
forecast_values = forecast[['ds','yhat']].values[:-12] # Use only the in-sample predictions

df2.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate accuracy metrics
mae = mean_absolute_error(actual_values, forecast_values)
mse = mean_squared_error(actual_values, forecast_values)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
