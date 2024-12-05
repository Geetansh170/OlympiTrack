import sqlite3
import pandas as pd
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt
from dbcrud import create_entry, delete_entry, read_entries, update_entry
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DB_FILE = "app/olympics_data.db"

def hypothesis1(st):
    Pre_Event_Results = read_entries("Pre_Event_Results")
    Pre_Event_Results['year'] = pd.to_numeric(Pre_Event_Results['year'], errors='coerce')
    Pre_Event_Results['participants'] = pd.to_numeric(Pre_Event_Results['participants'], errors='coerce')

    Pre_Event_Results = Pre_Event_Results.dropna(subset=['year', 'participants'])

    sports_multiple_years = Pre_Event_Results.groupby('sport').filter(lambda x: x['year'].nunique() > 1)

    sports_trend = sports_multiple_years.groupby('sport').apply(lambda df: linregress(df['year'], df['participants']).slope).reset_index()
    sports_trend.columns = ['sport', 'slope']

    sport_stats = sports_multiple_years.groupby('sport')['participants'].agg(['mean', 'std']).reset_index()
    sports_trend = sports_trend.merge(sport_stats, on='sport')

    X = sports_trend[['slope', 'mean']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    sports_trend['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_labels = {0: "rising", 1: "declining"} if sports_trend.groupby('cluster')['slope'].mean()[0] > 0 else {0: "declining", 1: "rising"}
    sports_trend['trend_label'] = sports_trend['cluster'].map(cluster_labels)

    silhouette_avg = silhouette_score(X_scaled, sports_trend['cluster'])
    davies_bouldin = davies_bouldin_score(X_scaled, sports_trend['cluster'])
    inertia = kmeans.inertia_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sports_trend, x='slope', y='mean', hue='trend_label', style='trend_label', palette='viridis', s=100)
    plt.title("Clustering of Sports by Rising and Declining Trends")
    plt.xlabel("Trend Slope")
    plt.ylabel("Average Participation")
    plt.grid(True)
    plt.legend(title='Trend')
    plt.show()

    st.pyplot(plt)

    sports_trend[['sport', 'slope', 'mean', 'std', 'trend_label']]
    st.dataframe(sports_trend)

    st.info(f"Silhouette Score: {silhouette_avg}")
    st.info(f"Davies-Bouldin Index: {davies_bouldin}")
    st.info(f"Inertia: {inertia}")

def hypothesis2(st):
    Pre_Athlete_Events_Details = read_entries("Pre_Athlete_Events_Details")
    Pre_Country_Profile = read_entries("Pre_Country_Profile")
    Pre_Population_Total = read_entries("Pre_Population_Total")


    Athlete_Events_Details_Mod = Pre_Athlete_Events_Details.merge(Pre_Country_Profile, left_on='country_noc', right_on='noc', how='left')
    Athlete_Events_Details_Mod.drop('noc', axis=1, inplace=True)
    Athlete_Events_Details_Mod.drop('country_noc', axis=1, inplace=True)
    print(Athlete_Events_Details_Mod.shape)
    print(Athlete_Events_Details_Mod.head(10))

    athlete_counts = Athlete_Events_Details_Mod.groupby(['country', 'year'])[['men', 'women']].sum().reset_index()
    athlete_counts = athlete_counts.rename(columns={'country': 'Country Name'})

    print(athlete_counts)

    Pre_Population_Total['Year'] = Pre_Population_Total['Year'].astype(int)
    athlete_counts['year'] = athlete_counts['year'].astype(int)
    merged_data = pd.merge(athlete_counts, Pre_Population_Total, left_on=['Country Name', 'year'],right_on=['Country Name', 'Year'], how='left')

    # print(merged_data)
    merged_data['percentage_women'] = (merged_data['women'] / merged_data['Count']) * 100

    merged_data['total'] = merged_data['men'] + merged_data['women']
    merged_data['percentage_women_better'] = (merged_data['women'] / merged_data['total']) * 100

    print(merged_data[['Country Name', 'year', 'percentage_women','percentage_women_better']].head(20))

    merged_data = merged_data.dropna(subset=['percentage_women'])

    min_val = merged_data['percentage_women'].min()
    max_val = merged_data['percentage_women'].max()
    merged_data['normalized_percentage_women'] = (merged_data['percentage_women'] - min_val) / (max_val - min_val)

    print(merged_data)
    merged_data = merged_data.dropna(subset=['percentage_women'])


    merged_data = merged_data.dropna(subset=['percentage_women_better'])
    average_participation = merged_data.groupby('Country Name')['percentage_women_better'].mean().reset_index()

    top_countries = average_participation.sort_values(by='percentage_women_better', ascending=False).head(20)
    df_filtered = merged_data[merged_data['Country Name'].isin(top_countries['Country Name'])]

    df_filtered = df_filtered[df_filtered['year'] >= 1960]
    train_data = df_filtered[df_filtered['year'] <= 2000]
    test_data = df_filtered[df_filtered['year'] > 2000]

    X_train = train_data[['year']]
    y_train = train_data['percentage_women_better']
    X_test = test_data[['year']]
    y_test = test_data['percentage_women_better']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    def create_dnn_model():
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(X_train_scaled.shape[1],)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    dnn_model = create_dnn_model()
    dnn_model.fit(X_train_scaled, y_train, epochs=150, batch_size=4, verbose=1)


    y_pred = dnn_model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    future_years = np.array([2024, 2028, 2032, 2036, 2040]).reshape(-1, 1)
    future_years_scaled = scaler.transform(future_years)
    future_pred = dnn_model.predict(future_years_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(train_data['year'], y_train, color='blue', label='Train Data')
    plt.scatter(test_data['year'], y_test, color='green', label='Actual Test Data')
    plt.scatter(test_data['year'], y_pred, color='red', label='Predicted Test Data')
    plt.plot(future_years, future_pred, 'o--', label='Future Predictions')
    plt.xlabel('Year')
    plt.ylabel('Percentage of Women Participation')
    plt.title('DNN Predictions for Women Participation (Top 50 Countries)')
    plt.legend()
    plt.grid(True)
    plt.show()

    st.pyplot(plt)

    abs_r2 = abs(r2)
    print(f"Absolute R² Score: {abs_r2:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Future Predictions (2024-2040): {future_pred.flatten()}")
    target_countries = [ 'Netherlands', 'Peru', 'Canada', 'Singapore', 'Romania','malta','angola']

    fig, axes = plt.subplots(len(target_countries), 1, figsize=(10, 4 * len(target_countries)), sharex=True)
    fig.suptitle('DNN Predictions for Women Participation by Country')

    for i, country in enumerate(target_countries):
        country_data = df_filtered[df_filtered['Country Name'].str.lower() == country.lower()]

        country_train_data = country_data[country_data['year'] <= 2000]
        country_test_data = country_data[country_data['year'] > 2000]

        X_train_country = country_train_data[['year']]
        y_train_country = country_train_data['percentage_women_better']
        X_test_country = country_test_data[['year']]
        y_test_country = country_test_data['percentage_women_better']
        X_train_country_scaled = scaler.transform(X_train_country)
        X_test_country_scaled = scaler.transform(X_test_country)

        y_pred_country = dnn_model.predict(X_test_country_scaled)



        future_pred_country = dnn_model.predict(future_years_scaled)



        ax = axes[i]
        ax.scatter(country_train_data['year'], y_train_country, color='blue', label='Train Data')
        ax.scatter(country_test_data['year'], y_test_country, color='green', label='Actual Test Data')
        ax.scatter(country_test_data['year'], y_pred_country, color='red', label='Predicted Test Data')
        ax.plot(future_years, future_pred_country, 'o--', label='Future Predictions')

        ax.set_title(f"{country}")
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of Women Participation')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    st.pyplot(plt)

    st.info(f"Absolute R² Score: {abs_r2:.3f}")
    st.info(f"Mean Absolute Error: {mae:.3f}")
    st.info(f"Future Predictions (2024-2040): {future_pred.flatten()}")

def hypothesis3(st):
    pass

def hypothesis4(st):
    pass