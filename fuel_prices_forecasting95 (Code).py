import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb 
from sklearn.metrics import mean_squared_error


df = pd.read_csv('C:\\Users\\aggel\Desktop\\Fuel_Price_Forecasting\\fuel_prices_forecasting95 (Dataset).csv',)

print(df.head())
print(df.tail())

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

print(df.index)
# Unleaded 95 oct. data visualization


color_pal = sns.color_palette()
df.plot(figsize=(15, 5), color='red', title='Fuel Average Price over time')
plt.ylim(bottom=0)
plt.ylim(top=2)
plt.show()

# End of visualization


# Train Test Split

train = df.loc[df.index < '28/02/2025']
test = df.loc[df.index >= '28/02/2025']

# Train test split visualization
fig, ax = plt.subplots(figsize=(15, 5))
train['Average_Price'].plot(ax=ax, label='Training Data', color='blue')
test['Average_Price'].plot(ax=ax, label='Test Data', color='red')
ax.axvline('28/02/2025', color = 'black', ls='--')
ax.legend(['Training Data', 'Test Data'])
ax.set_ylim(bottom=0)
ax.set_title('Fuel Price Over Time')
plt.show()

# one month of data plot
month_of_data = df.loc[(df.index > '25/10/2024') & (df.index <= '25/11/2024')]


fig, ax = plt.subplots(figsize=(15, 5))
month_of_data['Average_Price'].plot(ax=ax, color='blue')
ax.set_ylim(bottom=0)  
ax.set_title('month of data')
ax.set_ylim(top=2)
plt.show()
# two months

two_months_of_data = df.loc[(df.index > '25/10/2024') & (df.index <= '25/12/2024')]


fig, ax = plt.subplots(figsize=(15, 5))
two_months_of_data['Average_Price'].plot(ax=ax, color='blue')
ax.set_ylim(bottom=0)  
ax.set_title('two months of data')
ax.set_ylim(top=2)
plt.show()

# three months
three_months_of_data = df.loc[(df.index > '25/10/2024') & (df.index <= '25/01/2025')]


fig, ax = plt.subplots(figsize=(15, 5))
three_months_of_data['Average_Price'].plot(ax=ax, color='blue')
ax.set_ylim(bottom=0)  
ax.set_title('three months of data')
ax.set_ylim(top=2)
plt.show()

# four months

four_months_of_data = df.loc[(df.index > '25/10/2024') & (df.index <= '25/02/2025')]


fig, ax = plt.subplots(figsize=(15, 5))
four_months_of_data['Average_Price'].plot(ax=ax, color='blue')
ax.set_ylim(bottom=0)  
ax.set_title('four months of data')
ax.set_ylim(top=2)
plt.show()

# five months
five_months_of_data = df.loc[(df.index > '25/10/2024') & (df.index <= '25/03/2025')]

fig, ax = plt.subplots(figsize=(15, 5))
five_months_of_data['Average_Price'].plot(ax=ax, color='blue')
ax.set_ylim(bottom=0)  
ax.set_title('five months of data')
ax.set_ylim(top=2)
plt.show()

def create_features(df) :

 df = df.copy()
 df['hour'] = df.index.map(lambda x: x.hour)
 df['dayofweek'] = df.index.map(lambda x: x.dayofweek)
 df['quarter'] = df.index.map(lambda x: x.quarter)
 df['month'] = df.index.map(lambda x: x.month)
 df['dayofyear'] = df.index.map(lambda x: x.dayofyear)

 return df

df = create_features(df)

train = create_features(train)
test = create_features(test) 

X_train = train[['hour', 'dayofweek', 'quarter', 'month', 'dayofyear']]
y_train = train['Average_Price']

X_test = test[['hour', 'dayofweek', 'quarter', 'month', 'dayofyear']]
y_test = test['Average_Price']



X_train = pd.DataFrame(X_train)

y_train = pd.Series(y_train)

# create model 

reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                       learning_rate=0.1)
reg.fit(X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)


fi = pd.DataFrame(data=reg.feature_importances_, index=reg.feature_names_in_, columns=['importance'])

fi.sort_values('importance').plot(kind='barh', title='Feature importance')
plt.show()

# Forecast on test
predictions = reg.predict(X_test)
print(predictions)
print(test.columns)
test['prediction'] = reg.predict(X_test)


print(test[['Average_Price', 'prediction']])

# Merge the 'prediction' from test DataFrame into df DataFrame
result = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)


# Plot from the result DataFrame that contains both 'Average_Price' and 'prediction'
ax = result[['Average_Price']].plot(figsize=(15, 5))
result['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth data', 'Predictions'])
ax.set_title('Raw data and predictions')
ax.set_ylim(bottom=0)
plt.show()

score = np.sqrt(mean_squared_error(test['Average_Price'], test['prediction']))

print(f'Rmse score on test set :{score:0.2f}')

test['error'] = np.abs(test['Average_Price'] -test['prediction'])

print(test['error'])

# best and worst predictions

test['date'] = test.index.date 
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(5)

print("Worst predictions :" )
print(test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(5))

# best predictions 

test.groupby(['date'])['error'].mean().sort_values(ascending=True).head(5)

print("Best predictions : ")
print(test.groupby(['date'])['error'].mean().sort_values(ascending=True).head(5))

print(predictions)

# Predict the average price of Unleaded 95 gas in Greece on 4th of April 2025.
date_to_predict = pd.Timestamp('2025-04-04')


features = pd.DataFrame({
    'hour': [date_to_predict.hour],
    'dayofweek': [date_to_predict.dayofweek],
    'quarter': [date_to_predict.quarter],
    'month': [date_to_predict.month],
    'dayofyear': [date_to_predict.dayofyear]
})


predicted_price = reg.predict(features)

print(f"Predicted Average Price on 04/04/2025 : ")
print(predicted_price)

## END ## 
