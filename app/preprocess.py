import sqlite3
import pandas as pd
import pandas as pd
from dbcrud import read_entries

DB_FILE = "app/olympics_data.db"

def preprocess_data():

    Athlete_Events_Details = read_entries("Athlete_Events_Details")
    Event_Results = read_entries("Event_Results")
    Athlete_Biography = read_entries("Athlete_Biography")
    Population_Total = read_entries("Population_Total")
    Country_Profile = read_entries("Country_Profile")

    Pre_Event_Results =  Event_Results.copy()
    Pre_Event_Results.drop_duplicates(inplace=True)
    Pre_Event_Results.drop(['sport_url','result_date','result_location','result_format','result_detail','result_description'], axis=1, inplace=True)
    Pre_Event_Results['event_title'] = Pre_Event_Results['event_title'].str.strip().str.lower()
    Pre_Event_Results['edition'] = Pre_Event_Results['edition'].str.strip().str.lower()
    Pre_Event_Results['sport'] = Pre_Event_Results['sport'].str.strip().str.lower()
    Pre_Event_Results['result_participants'] = Pre_Event_Results['result_participants'].str.strip().str.lower()
    Pre_Event_Results[['participants', 'participant_countries']] = Pre_Event_Results['result_participants'].str.extract(r'(\d+)\sfrom\s(\d+)')
    Pre_Event_Results['participants'] = Pre_Event_Results['participants'].astype(int)
    Pre_Event_Results['participant_countries'] = Pre_Event_Results['participant_countries'].astype(int)
    Pre_Event_Results.drop('result_participants', axis=1, inplace=True)
    Pre_Event_Results['men'] = Pre_Event_Results['event_title'].str.contains(r'\bmen\b', case=False).astype(int)
    Pre_Event_Results['women'] = Pre_Event_Results['event_title'].str.contains('women', case=False).astype(int)
    Pre_Event_Results[['year', 'olympic_type']] = Pre_Event_Results['edition'].str.extract(r'(\d{4})\s+(summer|winter)\s+olympics')
    Pre_Event_Results.dropna(subset=['event_title', 'sport', 'participants', 'participant_countries', 'men', 'women','year','olympic_type'], inplace=True)
    olympic_dummies = pd.get_dummies(Pre_Event_Results['olympic_type'])
    olympic_dummies = olympic_dummies.astype(int)
    Pre_Event_Results = pd.concat([Pre_Event_Results, olympic_dummies], axis=1)

    Pre_Population_Total = Population_Total.copy()
    Pre_Population_Total['Country Name'] = Pre_Population_Total['Country Name'].str.strip().str.lower()
    Pre_Population_Total.dropna(subset=['Country Name', 'Year', 'Count'], inplace=True)
    Pre_Population_Total['Year'] = Pre_Population_Total['Year'].replace(2017, 2020)

    Pre_Athlete_Biography = Athlete_Biography.copy()
    Pre_Athlete_Biography.drop_duplicates(inplace=True)
    Pre_Athlete_Biography.drop(['description','special_notes'], axis=1, inplace=True)
    Pre_Athlete_Biography['country'] = Pre_Athlete_Biography['country'].str.strip().str.lower()
    Pre_Athlete_Biography['sex'] = Pre_Athlete_Biography['sex'].str.strip().str.lower()
    Pre_Athlete_Biography['name'] = Pre_Athlete_Biography['name'].str.strip().str.lower()
    Pre_Athlete_Biography['born'] = Pre_Athlete_Biography['born'].str.strip().str.lower()

    Pre_Athlete_Events_Details = Athlete_Events_Details.copy()
    Pre_Athlete_Events_Details = Pre_Athlete_Events_Details.drop_duplicates()
    Pre_Athlete_Events_Details = Pre_Athlete_Events_Details.drop(columns=['result_id', 'athlete', 'pos'])
    Pre_Athlete_Events_Details['medal'].fillna('no medal', inplace=True)
    Pre_Athlete_Events_Details['country_noc'] = Pre_Athlete_Events_Details['country_noc'].str.strip().str.lower()
    Pre_Athlete_Events_Details['sport'] = Pre_Athlete_Events_Details['sport'].str.strip().str.lower()
    Pre_Athlete_Events_Details['event'] = Pre_Athlete_Events_Details['event'].str.strip().str.lower()
    Pre_Athlete_Events_Details['edition'] = Pre_Athlete_Events_Details['edition'].str.strip().str.lower()
    Pre_Athlete_Events_Details['edition'] = Pre_Athlete_Events_Details['edition'].astype('category')
    Pre_Athlete_Events_Details['isTeamSport'] = Pre_Athlete_Events_Details['isTeamSport'].astype(bool)
    Pre_Athlete_Events_Details[['year', 'olympic_type']] = Pre_Athlete_Events_Details['edition'].str.extract(r'(\d{4})\s+(summer|winter)\s+olympics')
    Pre_Athlete_Events_Details['men'] = Pre_Athlete_Events_Details['event'].str.contains(r'\bmen\b', case=False).astype(int)
    Pre_Athlete_Events_Details['women'] = Pre_Athlete_Events_Details['event'].str.contains('women', case=False).astype(int)

    Pre_Country_Profile = Country_Profile.copy()
    Pre_Country_Profile['noc'] = Pre_Country_Profile['noc'].str.strip().str.lower()
    Pre_Country_Profile['country'] = Pre_Country_Profile['country'].str.strip().str.lower()

    with sqlite3.connect(DB_FILE) as conn:
        Pre_Event_Results.to_sql('Pre_Event_Results', conn, if_exists='replace', index=False)
        Pre_Athlete_Biography.to_sql('Pre_Athlete_Biography', conn, if_exists='replace', index=False)
        Pre_Population_Total.to_sql('Pre_Population_Total', conn, if_exists='replace', index=False)
        Pre_Country_Profile.to_sql('Pre_Country_Profile', conn, if_exists='replace', index=False)
        Pre_Athlete_Events_Details.to_sql('Pre_Athlete_Events_Details', conn, if_exists='replace', index=False)






