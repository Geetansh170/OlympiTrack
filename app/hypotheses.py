import sqlite3
import pandas as pd
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt
from dbcrud import create_entry, delete_entry, read_entries, update_entry

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

    sports_trend[['sport', 'slope', 'mean', 'std', 'trend_label']]
    print(sports_trend)

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"Inertia: {inertia}")

def hypothesis2(st):
    pass

def hypothesis3(st):
    pass

def hypothesis4(st):
    pass