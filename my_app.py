import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('rossmann-store-sales/clean_data.csv')
store_df = pd.read_csv('rossmann-store-sales/store.csv')

# Load model and preprocessors
model = joblib.load('models/XGB_Model.pkl')
scaler = joblib.load('models/standard_scaler.pkl')
pca = joblib.load('models/pca_transformer.pkl')
x_train_columns = joblib.load('models/x_train_columns.pkl')
# Note: Imputation is skipped as input data has no missing values

# Use the exact feature columns from training
feature_columns = x_train_columns

# App title
st.title("Rossmann Store Sales Dashboard")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose Section", ["Data Visualizations", "Sales Prediction"])

if option == "Data Visualizations":
    st.header("Data Visualizations")

    # Visualization 1: Monthly total sales by store type
    st.subheader("Monthly Sales by Store Type Over Time")
    monthly_store_sales = df.groupby(['Year', 'Month', 'StoreType'])['Sales'].sum().reset_index()
    monthly_store_sales['Year-Month'] = monthly_store_sales['Year'].astype(str) + '-' + monthly_store_sales['Month'].astype(str)
    fig1 = px.line(monthly_store_sales, x='Year-Month', y='Sales', color='StoreType', markers=True,
                   title='Monthly Sales by Store Type Over Time')
    st.plotly_chart(fig1)

    # Visualization 2: Monthly total sales overall
    st.subheader("Total Sales Over Time (Monthly)")
    monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
    monthly_sales['Year-Month'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str)
    fig2 = px.line(monthly_sales, x='Year-Month', y='Sales', markers=True, title='Total Sales Over Time (Monthly)')
    st.plotly_chart(fig2)

    # Visualization 3: Monthly average sales
    st.subheader("Average Sales Over Time (Monthly)")
    monthly_avg_sales = df.groupby(['Year', 'Month'])['Sales'].mean().reset_index()
    monthly_avg_sales['Year-Month'] = monthly_avg_sales['Year'].astype(str) + '-' + monthly_avg_sales['Month'].astype(str)
    fig3 = px.line(monthly_avg_sales, x='Year-Month', y='Sales', markers=True, title='Average Sales Over Time (Monthly)', color_discrete_sequence=['orange'])
    st.plotly_chart(fig3)

    # Visualization 4: Sales trend by Promo2 status
    st.subheader("Sales Trend by Promo2 Status")
    monthly_promo2 = df.groupby(['Year', 'Month', 'Promo2'])['Sales'].sum().reset_index()
    monthly_promo2['Year-Month'] = monthly_promo2['Year'].astype(str) + '-' + monthly_promo2['Month'].astype(str)
    fig4 = px.line(monthly_promo2, x='Year-Month', y='Sales', color='Promo2', markers=True, title='Sales Trend by Promo2 Status')
    st.plotly_chart(fig4)

    # Visualization 5: Average sales by day of the week
    st.subheader("Average Sales by Day of Week")
    avg_sales_day = df.groupby('DayOfWeek')['Sales'].mean().reset_index()
    fig5, ax = plt.subplots(figsize=(8, 5))
    ax.bar(avg_sales_day['DayOfWeek'], avg_sales_day['Sales'], color='skyblue')
    ax.set_title('Average Sales by Day of Week')
    ax.set_xlabel('Day of Week (1=Monday)')
    ax.set_ylabel('Average Sales')
    ax.set_xticks(avg_sales_day['DayOfWeek'])
    st.pyplot(fig5)

    # Visualization 6: Sales distribution by store type
    st.subheader("Sales Distribution by Store Type")
    fig6, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='StoreType', y='Sales', palette='rocket', ax=ax)
    ax.set_title('Sales Distribution by Store Type')
    st.pyplot(fig6)

    # Visualization 7: Sales distribution by Promo
    st.subheader("Effect of Promo on Sales")
    fig7, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='Promo', y='Sales', palette='rocket', ax=ax)
    ax.set_title('Effect of Promo on Sales')
    st.pyplot(fig7)

    # Visualization 8: Average sales by state holiday
    st.subheader("Average Sales during State Holidays")
    avg_sales_holiday = df.groupby('StateHoliday')['Sales'].mean().reset_index()
    holiday_labels = {'0': 'No Holiday', 'a': 'Public Holiday', 'b': 'Easter Holiday', 'c': 'Christmas Holiday'}
    avg_sales_holiday['StateHoliday'] = avg_sales_holiday['StateHoliday'].map(holiday_labels)
    fig8 = px.bar(avg_sales_holiday, x='StateHoliday', y='Sales', color='StateHoliday', title='Average Sales during State Holidays',
                  color_discrete_sequence=px.colors.qualitative.Set2, text='Sales')
    fig8.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig8.update_layout(showlegend=False, height=500, width=800, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='white', font=dict(size=14), title_x=0.5)
    st.plotly_chart(fig8)

    # Visualization 9: Customers vs Sales
    st.subheader("Customers vs Sales")
    fig9 = px.scatter(df, x='Customers', y='Sales', title="Customers vs Sales", template='plotly_dark')
    fig9.update_traces(marker=dict(color='gold', size=7))
    st.plotly_chart(fig9)

    # Visualization 10: Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=['float64', 'int64', 'int32']).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig10, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='rocket_r', square=True, ax=ax)
    ax.set_title('Heatmap of Correlations Between Numerical Features')
    st.pyplot(fig10)

elif option == "Sales Prediction":
    st.header("Sales Prediction")

    # Input widgets for basic features
    store = st.selectbox("Store ID", options=list(store_df['Store'].unique()), index=0)
    date = st.date_input("Date", value=pd.to_datetime('2014-06-15'))
    open_store = st.selectbox("Is Store Open?", [1, 0], index=0)  # 1 for open
    promo = st.selectbox("Is there a Promotion?", [0, 1], index=0)
    state_holiday = st.selectbox("State Holiday", ['0', 'a', 'b', 'c'], index=0)
    school_holiday = st.selectbox("School Holiday", [0, 1], index=0)

    # Get store info
    store_info = store_df[store_df['Store'] == store].iloc[0]
    store_type = store_info['StoreType']
    assortment = store_info['Assortment']
    compdistance = store_info['CompetitionDistance']
    compmonth = store_info['CompetitionOpenSinceMonth']
    compyear = store_info['CompetitionOpenSinceYear']
    promo2 = store_info['Promo2']
    promo2_since_week = store_info['Promo2SinceWeek']
    promo2_since_year = store_info['Promo2SinceYear']
    promo_interval = store_info['PromoInterval']
    if pd.isna(promo_interval):
        promo_interval = 'noPromo'

    # Extract date features
    year = date.year
    month = date.month
    day = date.day
    day_name = date.strftime('%A')  # e.g., 'Monday'

    # Preprocess inputs
    input_data = pd.DataFrame({
        'Store': [store],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'StateHoliday': [state_holiday],
        'SchoolHoliday': [school_holiday],
        'Promo': [promo],
        'Promo2': [promo2],
        'Open': [open_store],
        'compdistance': [compdistance],
        'compmonth': [compmonth],
        'compyear': [compyear],
        'Promo2SinceWeek': [promo2_since_week],
        'Promo2SinceYear': [promo2_since_year],
        'Assortment': [assortment],
        'StoreType': [store_type],
        'PromoInterval': [promo_interval],
        'Day_name': [day_name]
    })

    # Impute missing values with medians from store_df
    numeric_cols_store = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']
    medians = store_df[numeric_cols_store].median()
    # Map to input_data columns
    input_data['compdistance'] = input_data['compdistance'].fillna(medians['CompetitionDistance'])
    input_data['compmonth'] = input_data['compmonth'].fillna(medians['CompetitionOpenSinceMonth'])
    input_data['compyear'] = input_data['compyear'].fillna(medians['CompetitionOpenSinceYear'])
    input_data['Promo2SinceWeek'] = input_data['Promo2SinceWeek'].fillna(medians['Promo2SinceWeek'])
    input_data['Promo2SinceYear'] = input_data['Promo2SinceYear'].fillna(medians['Promo2SinceYear'])

    # Encode categorical
    stateholiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    assortment_map = {'a': 0, 'b': 1, 'c': 2}
    input_data['StateHoliday'] = input_data['StateHoliday'].map(stateholiday_map)
    input_data['Assortment'] = input_data['Assortment'].map(assortment_map)

    # One-hot encode
    input_data = pd.get_dummies(input_data, columns=['StoreType', 'PromoInterval', 'Day_name'], dtype='int64', drop_first=True)

    # Ensure all columns are present
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_columns]

    # Scale and pca
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)

    # Predict
    prediction = model.predict(input_pca)[0]

    st.subheader("Predicted Sales")
    st.write(f"The predicted sales for the given inputs is: €{prediction:.2f}")

if __name__ == "__main__":
    st.write("Run this app with: streamlit run my_app.py")
