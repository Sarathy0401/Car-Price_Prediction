import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/sarus/Downloads/car_data.csv')
    
    # Encode categorical variables
    label_encoders = {}
    for col in ['Fuel', 'Seller_Type']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Store encoders for later use
    
    return data, label_encoders

# Train model
@st.cache_resource
def train_model(data):
    X = data[['Year', 'Mileage', 'Fuel', 'Seller_Type']]
    y = data['Selling_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2, X_train, y_train

# Main application
def main():
    st.title('🚗 Car Selling Price Prediction & Analysis')
    
    # Load data
    data, label_encoders = load_data()
    
    # Train model
    model, mae, r2, X_train, y_train = train_model(data)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Predict Price", "View Dataset", "Model Insights"])
    
    with tab1:
        st.header("Price Prediction")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input('Year', min_value=data['Year'].min(), max_value=data['Year'].max(), value=2020)
                mileage = st.number_input('Mileage', min_value=int(data['Mileage'].min()), value=int(data['Mileage'].median()))
            with col2:
                fuel = st.selectbox('Fuel Type', label_encoders['Fuel'].classes_)
                seller_type = st.selectbox('Seller Type', label_encoders['Seller_Type'].classes_)
            
            submitted = st.form_submit_button("Predict Price")
            if submitted:
                # Encode categorical inputs
                fuel_encoded = label_encoders['Fuel'].transform([fuel])[0]
                seller_encoded = label_encoders['Seller_Type'].transform([seller_type])[0]
                
                input_data = [[year, mileage, fuel_encoded, seller_encoded]]
                prediction = model.predict(input_data)[0]
                st.success(f'Predicted Selling Price: ${prediction:,.2f}')
    
    with tab2:
        st.header("Dataset Overview")
        st.dataframe(data.head(100), height=400)
        
        st.subheader("Basic Statistics")
        st.write(data.describe())
    
    with tab3:
        st.header("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error", f"${mae:,.2f}")
        col2.metric("R² Score", f"{r2:.2%}")
        
        st.subheader("Feature Relationships")
        feature = st.selectbox('Select feature to plot', ['Year', 'Mileage'])
        fig = px.scatter(data, x=feature, y='Selling_Price', trendline="ols")
        st.plotly_chart(fig)
        
        st.subheader("Feature Importance")
        coefficients = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.coef_
        }).sort_values('Importance', ascending=False)
        fig2 = px.bar(coefficients, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig2)

if __name__ == '__main__':
    main()
