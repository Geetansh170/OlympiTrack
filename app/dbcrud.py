import sqlite3
import pandas as pd

# Database connection
DB_FILE = "app/olympics_data.db"

def create_tables():
    """
    Create tables if they do not exist in the database.
    """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Athlete_Events_Details (
                edition INTEGER,
                edition_id INTEGER,
                country_noc TEXT,
                sport TEXT,
                event TEXT,
                result_id INTEGER,
                athlete TEXT,
                athlete_id INTEGER,
                pos INTEGER,
                medal TEXT,
                isTeamSport INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Event_Results (
                result_id INTEGER PRIMARY KEY,
                event_title TEXT,
                edition INTEGER,
                edition_id INTEGER,
                sport TEXT,
                sport_url TEXT,
                result_date TEXT,
                result_location TEXT,
                result_participants INTEGER,
                result_format TEXT,
                result_detail TEXT,
                result_description TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Athlete_Biography (
                athlete_id INTEGER PRIMARY KEY,
                name TEXT,
                sex TEXT,
                born TEXT,
                height REAL,
                weight REAL,
                country TEXT,
                country_noc TEXT,
                description TEXT,
                special_notes TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Medal_Tally (
                edition INTEGER,
                edition_id INTEGER,
                year INTEGER,
                country TEXT,
                country_noc TEXT,
                gold INTEGER,
                silver INTEGER,
                bronze INTEGER,
                total INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Games_Summary (
                edition INTEGER,
                edition_id INTEGER PRIMARY KEY,
                edition_url TEXT,
                year INTEGER,
                city TEXT,
                country_flag_url TEXT,
                country_noc TEXT,
                start_date TEXT,
                end_date TEXT,
                competition_date TEXT,
                isHeld INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Population_Total (
                country_name TEXT,
                year INTEGER,
                count INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Country_Profile (
                country TEXT,
                continent TEXT,
                gdp INTEGER,
                population INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Pre_Event_Results (
                result_id INTEGER PRIMARY KEY,
                event_title TEXT,
                edition INTEGER,
                edition_id INTEGER,
                sport TEXT,
                participants INTEGER,
                participant_countries TEXT,
                men INTEGER,
                women INTEGER,
                year INTEGER,
                olympic_type TEXT,
                summer INTEGER,
                winter INTEGER
            )
        ''')

        # Table for Pre_Population_Total
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Pre_Population_Total (
                country_name TEXT,
                year INTEGER,
                count INTEGER,
                PRIMARY KEY (country_name, year)
            )
        ''')

        # Table for Pre_Athlete_Biography
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Pre_Athlete_Biography (
                athlete_id INTEGER PRIMARY KEY,
                name TEXT,
                sex TEXT,
                born TEXT,
                height REAL,
                weight REAL,
                country TEXT,
                country_noc TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Pre_Athlete_Events_Details (
                edition INTEGER,
                edition_id INTEGER,
                country_noc TEXT,
                sport TEXT,
                event TEXT,
                athlete_id INTEGER,
                medal TEXT,
                isTeamSport INTEGER,
                year INTEGER,
                olympic_type TEXT,
                men INTEGER,
                women INTEGER,
                PRIMARY KEY (edition, edition_id, athlete_id, event)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Pre_Country_Profile (
                noc TEXT PRIMARY KEY,
                country TEXT
            )
        ''')

        conn.commit()
        print("Tables created successfully!")

def upload_data_in_db():
    """
    Uploads CSV files into the database.
    """
    with sqlite3.connect(DB_FILE) as conn:
        Athlete_Events_Details = pd.read_csv('app/csv_files/Olympic_Athlete_Event_Details.csv')
        Event_Results = pd.read_csv('app/csv_files/Olympic_Event_Results.csv')
        Athlete_Biography = pd.read_csv('app/csv_files/Olympic_Athlete_Biography.csv')
        Medal_Tally = pd.read_csv('app/csv_files/Olympic_Medal_Tally_History.csv')
        Games_Summary = pd.read_csv('app/csv_files/Olympic_Games_Summary.csv')
        Population_Total = pd.read_csv('app/csv_files/population_total_long.csv')
        Country_Profile = pd.read_csv('app/csv_files/Olympic_Country_Profiles.csv')

        Athlete_Events_Details.to_sql('Athlete_Events_Details', conn, if_exists='replace', index=False)
        Event_Results.to_sql('Event_Results', conn, if_exists='replace', index=False)
        Athlete_Biography.to_sql('Athlete_Biography', conn, if_exists='replace', index=False)
        Medal_Tally.to_sql('Medal_Tally', conn, if_exists='replace', index=False)
        Games_Summary.to_sql('Games_Summary', conn, if_exists='replace', index=False)
        Population_Total.to_sql('Population_Total', conn, if_exists='replace', index=False)
        Country_Profile.to_sql('Country_Profile', conn, if_exists='replace', index=False)

        print("Raw data has been uploaded successfully!")

# CRUD Functions
def create_entry(table, data):
    with sqlite3.connect(DB_FILE) as conn:
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        conn.execute(sql, list(data.values()))
        conn.commit()

def read_entries(table, condition=""):
    with sqlite3.connect(DB_FILE) as conn:
        sql = f"SELECT * FROM {table} {condition}"
        return pd.read_sql(sql, conn)

def update_entry(table, update_values, condition):
    with sqlite3.connect(DB_FILE) as conn:
        updates = ", ".join([f"{k} = ?" for k in update_values.keys()])
        sql = f"UPDATE {table} SET {updates} WHERE {condition}"
        conn.execute(sql, list(update_values.values()))
        conn.commit()

def delete_entry(table, condition):
    with sqlite3.connect(DB_FILE) as conn:
        sql = f"DELETE FROM {table} WHERE {condition}"
        conn.execute(sql)
        conn.commit()

def execute_query(query, params=()):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()

# Example Usage
if __name__ == "__main__":
    create_tables()
    # Upload data from CSV files to the database
    # upload_data_in_db()
