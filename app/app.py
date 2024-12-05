import streamlit as st
import pandas as pd
import sqlite3
import os

# Create SQLite database connection
conn = sqlite3.connect('data.db')

# Create a table for raw and preprocessed data if they don't exist
def create_tables():
    conn.execute('''
        CREATE TABLE IF NOT EXISTS raw_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS preprocessed_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT
        )
    ''')
create_tables()

# Function to save data to SQLite
def save_to_db(df, table_name):
    df.to_sql(table_name, conn, if_exists='replace', index=False)

# Function to load data from SQLite
def load_from_db(table_name):
    query = f'SELECT * FROM {table_name}'
    return pd.read_sql(query, conn)

# Function to preprocess data
def preprocess_data(df):
    df = df.fillna(0)  # Handle missing values by replacing with 0
    df = (df - df.min()) / (df.max() - df.min())  # Normalize data
    return df

st.title("Olympics Trends")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    # Read CSV and show preview
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(df.head())
    
    # Save raw data to SQLite
    save_to_db(df, "raw_data")
    st.success("Raw data saved to database!")

# Step 2: Preprocess Data
if st.button("Preprocess Data"):
    df_raw = load_from_db("raw_data")
    df_preprocessed = preprocess_data(df_raw)
    
    # Save preprocessed data to SQLite
    save_to_db(df_preprocessed, "preprocessed_data")
    st.success("Preprocessed data saved to database!")
    
    st.subheader("Preprocessed Data")
    st.write(df_preprocessed.head())

# Step 3: Display Stored Data
st.subheader("Raw Data from Database")
try:
    raw_data = load_from_db("raw_data")
    st.write(raw_data.head())
except Exception as e:
    st.write("No raw data available.")

st.subheader("Preprocessed Data from Database")
try:
    preprocessed_data = load_from_db("preprocessed_data")
    st.write(preprocessed_data.head())
except Exception as e:
    st.write("No preprocessed data available.")

# Step 4: Data Analysis Options
st.subheader("Data Analysis")
analysis_option = st.selectbox(
    "Choose an analysis:",
    ["Summary Statistics", "Correlation Matrix", "Data Distribution", "Missing Values Analysis"]
)

if analysis_option == "Summary Statistics":
    st.write("Summary Statistics of Preprocessed Data")
    st.write(preprocessed_data.describe())
    
elif analysis_option == "Correlation Matrix":
    st.write("Correlation Matrix of Preprocessed Data")
    st.write(preprocessed_data.corr())
    
elif analysis_option == "Data Distribution":
    st.write("Data Distribution")
    st.bar_chart(preprocessed_data)
    
elif analysis_option == "Missing Values Analysis":
    st.write("Missing Values Analysis")
    st.write(preprocessed_data.isnull().sum())

conn.close()
