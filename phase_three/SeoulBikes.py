# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:59:05 2024

@author: Marcin Krawczyk
"""
# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time



# function definition
def adjust_min(x,level):
    if x!=0:
        return int(x-level*abs(x))
    else:
        return 0

def adjust_max(x, level):
    if x!=0:
        return int(x+level*abs(x))
    else:
        return 0
    
def default_value(x_min, x_max, level, to_int=True):
    if to_int:
        return int(x_min + (x_max - x_min) * level)
    else:
        return x_min + (x_max - x_min) * level
        

if 'data_summary' not in st.session_state:
    st.session_state.data_summary = ''
if 'model' not in st.session_state:
    st.session_state.model = ''
if 'ref_cols' not in st.session_state:
    st.session_state.ref_cols = ''
if 'target' not in st.session_state:
    st.session_state.target = ''

if 'season_selected' not in st.session_state:
    st.session_state.season_selected = ''
if 'month_selected' not in st.session_state:
    st.session_state.month_selected = ''
# if 'week_selected' not in st.session_state:
#     st.session_state.week_selected = ''


# data and model load
if not isinstance(st.session_state.data_summary, pd.DataFrame):
    st.session_state.data_summary = pd.read_pickle("./dataset/data_summary.pkl")
    st.success('Data summary successfully loaded', icon="✅")

if not isinstance(st.session_state.ref_cols, list):
    st.session_state.model, st.session_state.ref_cols, st.session_state.target = joblib.load("./dataset/ml_model.pkl")
    st.success('Predictive model successfully loaded', icon="✅")


st.title('Seoul Bikes demand prediction')


col11, col12, col13, col14 = st.columns(4)

with col11:
    # Season selection
    seasons_list = st.session_state.data_summary['Season'].unique().tolist()
    st.session_state.season_selected = st.radio('**Select year season**', seasons_list, horizontal=False)

with col12:
    # Month selection
    months_list = st.session_state.data_summary[st.session_state.data_summary['Season'] == st.session_state.season_selected]['Month'].tolist()
    st.session_state.month_selected = st.radio('**Select year month**', months_list, horizontal=False)

with col13:
    # Holiday selection
    holiday_selected = st.radio('**Is it holiday?**', ['No holiday', 'Holiday'], horizontal=False)

with col14:
    # Daytype selection
    daytype_selected = st.radio('**Is it weekend?**', ['Weekday', 'Weekend'], horizontal=False)


col51, col52 = st.columns(2)

with col51:
    # Week selection
    week_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Week min'].iloc[0]
    week_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Week max'].iloc[0]
#    st.session_state.week_selected = st.slider('Select week number', min_value=week_min, max_value=week_max)
    week_selected = st.slider('**Select week number**', min_value=week_min, max_value=week_max, value = week_min+1)
    
with col52:
    # Hour selection
    hour_selected = st.slider('**Select the hour**', min_value=0, max_value=23, value=12)

col21, col22, col23 = st.columns(3)

with col21:
    # Temperature selection
    temp_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Temperature(°C) min'].item()
    temp_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Temperature(°C) max'].item()
    temperature_selected = st.slider('**Temperature**', min_value=adjust_min(temp_min, 0.2), max_value=adjust_max(temp_max, 0.2), 
                                     value = default_value(temp_min, temp_max, 0.25))

with col22:
    # Solar Radiation (MJ/m2) selection
    solar_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Solar Radiation (MJ/m2) min'].item()
    solar_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Solar Radiation (MJ/m2) max'].item()
    solar_selected = st.slider('**Solar Radiation (MJ/m2)**', min_value=solar_min, max_value=solar_max, 
                               value = default_value(solar_min, solar_max, 0.4, False))

with col23:
    # Visiility selection
    vis_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Visibility (10m) min'].item()
    vis_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Visibility (10m) max'].item()
    visibility_selected = st.slider('**Visibility (10m)**', min_value=adjust_min(vis_min,0), max_value=adjust_max(vis_max,0.2), 
                                    value = default_value(vis_min, vis_max, 0.8), step=100)

col31, col32, col33 = st.columns(3)

with col31:
    # Wind speed selection
    wind_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Wind speed (m/s) min'].item()
    wind_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Wind speed (m/s) max'].item()
    wind_selected = st.slider('**Wind speed (m/s)**', min_value=adjust_min(wind_min,0), max_value=adjust_max(wind_max,0.2),
                              value = default_value(wind_min, wind_max, 0.25))

col41, col42, col43 = st.columns(3)

with col41:
    # Rainfall(mm) selection
    rain_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Rainfall(mm) min'].item()
    rain_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Rainfall(mm) max'].item()
    rainfall_selected = st.slider('**Rainfall(mm)**', min_value=adjust_min(rain_min,0), max_value=adjust_max(rain_max,0.2))

with col42:
    # Humidity selection
    hum_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Humidity(%) min'].item()
    hum_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Humidity(%) max'].item()
    humidity_selected = st.slider('**Humidity(%)**', min_value=adjust_min(hum_min,0), max_value=adjust_max(hum_max,0),
                                  value = default_value(hum_min, hum_max, 0.25), step=5)

with col43:
    # Snowfall (cm) selection
    snow_min = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Snowfall (cm) min'].item()
    snow_max = st.session_state.data_summary[st.session_state.data_summary['Month'] == st.session_state.month_selected]['Snowfall (cm) max'].item()
    if snow_max!=0:
        snowfall_selected = st.slider('**Snowfall (cm)**', min_value=adjust_min(snow_min,0), max_value=adjust_max(snow_max,0.2))
    else:
        snowfall_selected = 0


def predict():
    row = np.array([hour_selected, temperature_selected, humidity_selected, wind_selected, 
                    visibility_selected, solar_selected, rainfall_selected, snowfall_selected,
                    st.session_state.season_selected, holiday_selected, daytype_selected, week_selected])
    X_new = pd.DataFrame([row], columns=st.session_state.ref_cols)
    prediction = st.session_state.model.predict(X_new)[0]
    return prediction
 

btn1 = st.button('Make a prediction', type='primary')
if btn1:
    prediction = predict()
    
    st.write('Predicted bikes demand:', round(prediction) )

# data_summary.info()
# st.info(st.session_state.ref_cols, icon="ℹ️")