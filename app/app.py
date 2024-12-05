import streamlit as st
import pandas as pd
import sqlite3

from dbcrud import create_entry, delete_entry, read_entries, update_entry, upload_data_in_db

DB_FILE = "app/olympics_data.db"

# Function to connect to the database and execute queries
def execute_query(query, params=()):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()

def display_and_modify_table(table_name):
    st.write(f"### Table: {table_name}")
    df = read_entries(table_name)
    st.dataframe(df)

    with st.expander("Add New Entry"):
        with st.form(f"add_entry_{table_name}"):
            new_entry = {}
            for column in df.columns:
                new_entry[column] = st.text_input(f"{column}", "")
            submitted = st.form_submit_button("Add Entry")
            if submitted:
                create_entry(table_name, new_entry)
                st.success("Entry added successfully!")

    with st.expander("Update Entry"):
        with st.form(f"update_entry_{table_name}"):
            condition = st.text_input("Update Condition (e.g., id = 1)")
            update_data = {}
            for column in df.columns:
                update_data[column] = st.text_input(f"New Value for {column}", "")
            submitted = st.form_submit_button("Update Entry")
            if submitted:
                update_entry(table_name, update_data, condition)
                st.success("Entry updated successfully!")

    with st.expander("Delete Entry"):
        condition = st.text_input(f"Delete Condition for {table_name} (e.g., id = 1)")
        if st.button(f"Delete Entry from {table_name}"):
            delete_entry(table_name, condition)
            st.success("Entry deleted successfully!")

st.title("Olympic Trends Analysis")

st.sidebar.header("Database Options")
if st.sidebar.button("Upload Data from Raw CSV Files"):
    upload_data_in_db()

st.sidebar.warning("This will overwrite all existing data.")

# Display available tables
st.sidebar.header("Available Tables")
tables = ["Athlete_Events_Details", "Event_Results", "Athlete_Biography",
          "Medal_Tally", "Games_Summary", "Population_Total", "Country_Profile"]

selected_table = st.sidebar.selectbox("Select Table to View/Modify", tables)
if selected_table:
    display_and_modify_table(selected_table)

st.header("Olympic Trends Analysis")
st.write("""
### Hypotheses
1. **Countries with higher GDP tend to win more medals.**
2. **Host countries tend to perform better in terms of medal tally.**
3. **Athletes from countries with higher populations have a higher chance of reaching finals.**
4. **Participation in team sports correlates with stronger performance in individual events.**
""")
st.info("Analysis for these hypotheses will be implemented soon.")
