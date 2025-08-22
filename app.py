import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

#Load trained RandomForest model
model = pickle.load(open("best_taxi_fare_model.pkl", "rb"))

st.title("NYC Taxi Fare Prediction🚖")
st.write("Enter the trip details to predict the fare.")

#User Inputs
pickup_date = st.date_input("Pickup Date", datetime(2025, 1, 1))
pickup_time = st.time_input("Pickup Time", datetime(2025, 1, 1, 12, 0).time())
pickup_longitude = st.number_input("Pickup Longitude", value=-73.9857)
pickup_latitude = st.number_input("Pickup Latitude", value=40.7484)
dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.9851)
dropoff_latitude = st.number_input("Dropoff Latitude", value=40.7549)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
mta_tax = st.number_input("MTA Tax", value=0.5)
tip_amount = st.number_input("Tip Amount", value=1.0)
tolls_amount = st.number_input("Tolls Amount", value=0.0)
improvement_surcharge = st.number_input("Improvement Surcharge", value=0.3)

vendor_id = st.selectbox("VendorID", [1, 2])
rate_code_id = st.selectbox("Rate Code ID", [1, 2, 3, 4, 5, 99])
store_and_fwd_flag = st.selectbox("Store and Forward Flag", ["N", "Y"])
payment_type = st.selectbox("Payment Type", [1, 2, 3, 4])  #1=Credit, 2=Cash, etc.

#Feature Engineering
pickup_datetime = datetime.combine(pickup_date, pickup_time)

#Datetime features
year = pickup_datetime.year
month = pickup_datetime.month
day = pickup_datetime.day
day_of_week = pickup_datetime.weekday()   #Monday=0
hour = pickup_datetime.hour
is_weekend = 1 if day_of_week >= 5 else 0
am_pm = 0 if hour < 12 else 1  #0 = AM, 1 = PM

#Distance calculation (Haversine)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

trip_distance = haversine(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

#Derived features
fare_per_km = 0 if trip_distance == 0 else (tip_amount + tolls_amount + improvement_surcharge) / trip_distance
fare_per_min = 0  #we don't know trip duration → keep 0

#One-hot encodings
vendor_2 = 1 if vendor_id == 2 else 0
ratecode_2 = 1 if rate_code_id == 2 else 0
ratecode_3 = 1 if rate_code_id == 3 else 0
ratecode_4 = 1 if rate_code_id == 4 else 0
ratecode_5 = 1 if rate_code_id == 5 else 0
ratecode_99 = 1 if rate_code_id == 99 else 0
store_and_fwd_flag_Y = 1 if store_and_fwd_flag == "Y" else 0
payment_type_2 = 1 if payment_type == 2 else 0
payment_type_3 = 1 if payment_type == 3 else 0
payment_type_4 = 1 if payment_type == 4 else 0

#Build feature DataFrame
features = pd.DataFrame([{
    "passenger_count": passenger_count,
    "pickup_longitude": pickup_longitude,
    "pickup_latitude": pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude": dropoff_latitude,
    "mta_tax": mta_tax,
    "tip_amount": tip_amount,
    "tolls_amount": tolls_amount,
    "improvement_surcharge": improvement_surcharge,
    "year": year,
    "month": month,
    "day": day,
    "day_of_week": day_of_week,
    "hour": hour,
    "trip_distance": trip_distance,
    "is_weekend": is_weekend,
    "am_pm": am_pm,
    "fare_per_km": fare_per_km,
    "fare_per_min": fare_per_min,
    "VendorID_2": vendor_2,
    "RatecodeID_2": ratecode_2,
    "RatecodeID_3": ratecode_3,
    "RatecodeID_4": ratecode_4,
    "RatecodeID_5": ratecode_5,
    "RatecodeID_99": ratecode_99,
    "store_and_fwd_flag_Y": store_and_fwd_flag_Y,
    "payment_type_2": payment_type_2,
    "payment_type_3": payment_type_3,
    "payment_type_4": payment_type_4
}])

#Ensure exact order of columns (same as training)
features = features[[
    'passenger_count', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'mta_tax', 'tip_amount',
    'tolls_amount', 'improvement_surcharge', 'year', 'month', 'day',
    'day_of_week', 'hour', 'trip_distance', 'is_weekend', 'am_pm',
    'fare_per_km', 'fare_per_min', 'VendorID_2', 'RatecodeID_2',
    'RatecodeID_3', 'RatecodeID_4', 'RatecodeID_5', 'RatecodeID_99',
    'store_and_fwd_flag_Y', 'payment_type_2', 'payment_type_3',
    'payment_type_4'
]]

#Prediction
if st.button("Predict Fare"):
    prediction = model.predict(features)[0]
    st.success(f"💰 Predicted Taxi Fare: **${prediction:.2f}**")