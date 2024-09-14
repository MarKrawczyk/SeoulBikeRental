### IMPORT LIBRARIES ###
import pandas as pd

# import the dataset
data = pd.read_csv('../dataset/SeoulBikeData.csv', encoding='Windows-1252')

# drop two columns "Functioning Day" and "Dew point temperature°C"
data = data.drop(columns=['Functioning Day', 'Dew point temperature(°C)'])

# converting the 'Date' column to 'datetime'
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Feature engineering
data['DayType'] = data['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
data['WeekNumber'] = data['Date'].dt.isocalendar().week
data['Month'] = data['Date'].dt.month_name()
data.drop(columns=['Date'], inplace=True)

# conert non-numerical columns to dtype category
categorical_fetures = [
    "Seasons",
    "Holiday",
    "DayType"
]
data[categorical_fetures] = data[categorical_fetures].astype("category")

data.info()
data['Seasons'].unique()

# data aggregation on Month
monthly_stats = data.groupby('Month').agg({
    'Temperature(°C)': ['min', 'max'],
    'Humidity(%)': ['min', 'max'],
    'Wind speed (m/s)': ['min', 'max'],
    'Visibility (10m)': ['min', 'max'],
    'Solar Radiation (MJ/m2)': ['min', 'max'],
    'Rainfall(mm)': ['min', 'max'],
    'Snowfall (cm)': ['min', 'max']    
}).reset_index()

data = data.drop(columns=['Month'])


data.to_pickle("../dataset/processed_data.pkl")


months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

seasons = ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 
           'Summer', 'Summer', 'Autumn', 'Autumn', 'Autumn', 'Winter']

week_min = [1, 6, 10, 14, 19, 23, 27, 32, 36, 40, 45, 49]

week_max = [5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48, 52]

# New dataframe with summary 
data_summary = pd.DataFrame({
    'Season': seasons,
    'Month': months,
    'Week min': week_min,
    'Week max': week_max
})

# Płaskie nazwy kolumn po agregacji
monthly_stats.columns = ['Month',
                         'Temperature(°C) min', 'Temperature(°C) max',
                         'Humidity(%) min', 'Humidity(%) max',
                         'Wind speed (m/s) min', 'Wind speed (m/s) max',
                         'Visibility (10m) min', 'Visibility (10m) max',
                         'Solar Radiation (MJ/m2) min', 'Solar Radiation (MJ/m2) max',
                         'Rainfall(mm) min', 'Rainfall(mm) max',
                         'Snowfall (cm) min', 'Snowfall (cm) max',]

# Teraz łączymy te dane z istniejącym DataFrame df na podstawie kolumny 'Month'
data_summary = pd.merge(data_summary, monthly_stats, on='Month', how='left')

data_summary.to_pickle("../dataset/data_summary.pkl")


data["Temperature(°C)"].describe()
