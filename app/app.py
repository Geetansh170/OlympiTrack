import streamlit as st
from hypotheses import hypothesis1, hypothesis2, hypothesis3, hypothesis4
from streamlit_helper import display_and_modify_table
from preprocess import preprocess_data
from dbcrud import upload_data_in_db

st.set_page_config(
    page_title="Olympic Trends Analysis",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="auto"
)

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
st.warning("This will truncate Preprocessed tables")

if st.button("Run Preprocessing"):
    preprocess_data()

st.header("Available Processed Tables")
tables = ["Pre_Event_Results", "Pre_Population_Total", "Pre_Athlete_Biography", "Pre_Athlete_Events_Details", "Pre_Country_Profile"]

selected_table = st.selectbox("Select Table to View/Modify", tables)
if selected_table:
    display_and_modify_table(st, selected_table)

st.header("Olympic Trends Analysis")
st.write("""
### Hypotheses
1. **What is the general trend in women participation country wise over the years? What countries are doing well and how do they compare to the best performing countries?**
2. **Are there any sports which are on the decline and losing popularity among participants? Also, are there some sports which have gained popularity over the recent years?**
3. **How do the trends in medal counts for team sports compare to those for individual sports across different countries over the years, and what insights can be drawn from these comparisons regarding each country's performance in the Olympic Games?**
4. **In athletics, height, weight, age, and country are major indicators of success in the Olympics. We have made efforts to achieve the same.**
""")

st.header("Hypothesis 1")
if st.button("Run Hypothesis 1"):
     hypothesis1(st)

st.header("Hypothesis 2")
if st.button("Run Hypothesis 2"):
     hypothesis2(st)

st.header("Hypothesis 3")
if st.button("Run Hypothesis 3"):
     hypothesis3(st)

st.header("Hypothesis 4")
if st.button("Run Hypothesis 4"):
     hypothesis4(st)
