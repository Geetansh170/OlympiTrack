import streamlit as st
from streamlit_helper import display_and_modify_table
from preprocess import preprocess_data
from dbcrud import upload_data_in_db

st.title("Olympic Trends Analysis")

st.header("Upload from Raw CSV files")
st.warning("This will overwrite all existing data if its exists otherwise will just populate tables. Do it if you have think your tables have been modified incorrectly.")
if st.button("Upload"):
    st.info("Uploading")
    upload_data_in_db()
    st.info("Data has been successfully populated in Raw Tables")

st.header("Available raw tables")
tables = ["Athlete_Events_Details", "Event_Results", "Athlete_Biography",
          "Medal_Tally", "Games_Summary", "Population_Total", "Country_Profile"]

selected_table = st.selectbox("Select Table to View/Modify", tables)
if selected_table:
    display_and_modify_table(st, selected_table)

st.header("Preprocessing Steps")
st.info("These will do preprocessing from Raw Data in database and also create new tables.")
st.warning("This will truncate Preprcoessed tables")

if st.button("Run Preprocessing"):
    preprocess_data()

st.header("Preprocessed available Tables")
tables = ["Pre_Event_Results", "Pre_Population_Total", "Pre_Athlete_Biography", "Pre_Athlete_Events_Details", "Pre_Country_Profile"]

selected_table = st.selectbox("Select Table to View/Modify", tables)
if selected_table:
    display_and_modify_table(st, selected_table)

st.header("Olympic Trends Analysis")
st.write("""
### Hypotheses
1. **Countries with higher GDP tend to win more medals.**
2. **Host countries tend to perform better in terms of medal tally.**
3. **Athletes from countries with higher populations have a higher chance of reaching finals.**
4. **Participation in team sports correlates with stronger performance in individual events.**
""")
st.info("Analysis for these hypotheses will be implemented soon.")
