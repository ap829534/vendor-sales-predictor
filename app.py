import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("dynamic_vendors_sales_data.csv")
df['Month'] = pd.to_datetime(df['Date']).dt.month

# Add percentage-based labels
def decide_label(row):
    sold_pct = row['Sold(Kg)'] / row['Stock(Kg)'] if row['Stock(Kg)'] > 0 else 0
    unsold_pct = row['Unsold(Kg)'] / row['Stock(Kg)'] if row['Stock(Kg)'] > 0 else 0
    spoiled_pct = row['Spoiled(Kg)'] / row['Stock(Kg)'] if row['Stock(Kg)'] > 0 else 0

    if spoiled_pct > sold_pct and spoiled_pct > unsold_pct:
        return 'Spoiled'
    elif unsold_pct > sold_pct and unsold_pct > spoiled_pct:
        return 'Unsold'
    else:
        return 'Sold'

df['Label'] = df.apply(decide_label, axis=1)

# Prepare features and target
features = ['VendorName', 'Product', 'Month', 'TotalAmount', 'Stock(Kg)', 'Spoiled(Kg)']
X = df[features].copy()
y = df['Label']

# Label encoding
encoders = {}
for col in ['VendorName', 'Product']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred, labels=['Sold', 'Unsold', 'Spoiled'])

# Streamlit UI
st.set_page_config(page_title="Vendor Sales Outcome Predictor", layout="wide")
st.title(" Vendor Sales Outcome Predictor")
st.markdown("""
Welcome to the **interactive prediction tool** for analyzing vendor performance. Enter the vendor details and sales info to predict the outcome.
""")

# Layout
col1, col2 = st.columns(2)
with col1:
    vendor = st.selectbox(" Select Vendor", df['VendorName'].unique())
    product = st.selectbox(" Select Product", df['Product'].unique())
    stock = st.number_input(" Enter Stock (Kg)", min_value=0.0, step=1.0)
    spoiled = st.number_input(" Enter Spoiled (Kg)", min_value=0.0, step=1.0)
with col2:
    month = st.selectbox(" Select Month", sorted(df['Month'].unique()))
    total_amount = st.number_input(" Enter Total Sales Amount", min_value=0.0, step=10.0)

# Predict button
input_data = pd.DataFrame({
    'VendorName': [encoders['VendorName'].transform([vendor])[0]],
    'Product': [encoders['Product'].transform([product])[0]],
    'Month': [month],
    'TotalAmount': [total_amount],
    'Stock(Kg)': [stock],
    'Spoiled(Kg)': [spoiled]
})

if st.button(" Predict Outcome"):
    prediction = model.predict(input_data)[0]
    st.success(f" Predicted Outcome: **{prediction}**")

# Display metrics
st.subheader(" Model Performance Metrics")
col3, col4, col5 = st.columns(3)
col3.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
col4.metric("Precision (Sold)", f"{report['Sold']['precision']*100:.2f}%")
col5.metric("Recall (Sold)", f"{report['Sold']['recall']*100:.2f}%")



# Charts
st.subheader(" Data Overview")
st.plotly_chart(px.histogram(df, x='VendorName', color='Label', barmode='group', title='Vendor Outcome Distribution'))
st.plotly_chart(px.pie(df, names='Label', title='Overall Outcome Share'))

