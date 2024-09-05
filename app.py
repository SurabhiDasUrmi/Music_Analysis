from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pycountry
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pycountry
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import geopandas as gpd
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__, template_folder='./templates',
            static_folder='./static')

file_path = './dataset/universal_top_spotify_songs.csv'
spotify_data = pd.read_csv(file_path)


# Normalize the audio features
features = ['tempo', 'energy', 'valence', 'loudness', 'danceability', 'speechiness', 'liveness']
scaler = MinMaxScaler()
spotify_data[features] = scaler.fit_transform(spotify_data[features])

spotify_data['positivity_score'] = (spotify_data['tempo'] + 
                                           spotify_data['energy'] + 
                                           spotify_data['loudness'] + 
                                            spotify_data['valence'] +
                                           spotify_data['danceability'] - 
                                           spotify_data['speechiness'] - 
                                           spotify_data['liveness'])

# Filter and process Europe data
europe_data = spotify_data[
    (spotify_data['snapshot_date'].str.startswith('2023')) |
    (spotify_data['snapshot_date'].str.startswith('2024'))
]
europe_data = europe_data[europe_data['country'].isin([
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU',
    'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
])]

# Map ISO codes to country names
iso_to_country = {country.alpha_2: country.name for country in pycountry.countries}
europe_data['country_full'] = europe_data['country'].map(iso_to_country)

# Aggregate features by country
aggregated_features = europe_data.groupby('country_full')[features].mean().reset_index()

# Normalize features
aggregated_features[features] = scaler.fit_transform(aggregated_features[features])

# Calculate positivity score
aggregated_features['positivity_score'] = (
    aggregated_features['tempo'] +
    aggregated_features['energy'] +
    aggregated_features['loudness'] +
    aggregated_features['valence'] +
    aggregated_features['danceability'] -
    aggregated_features['speechiness'] -
    aggregated_features['liveness']
)

# Rank countries
aggregated_features['positivity_rank'] = aggregated_features['positivity_score'].rank(ascending=False)
ranked_countries = aggregated_features.sort_values('positivity_rank')

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route to get correlation matrix data
@app.route('/api/correlation')
def get_correlation():
    corr_matrix = spotify_data[features].corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='darkmint')
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

# Route to get positivity scores for EU countries
@app.route('/api/positivity')
def get_positivity():
    europe_data = spotify_data[
        (spotify_data['snapshot_date'].str.startswith('2023')) |
        (spotify_data['snapshot_date'].str.startswith('2024'))
    ]
    europe_data = europe_data[europe_data['country'].isin([
        'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU',
        'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
    ])]

    iso_to_country = {country.alpha_2: country.name for country in pycountry.countries}
    europe_data['country_full'] = europe_data['country'].map(iso_to_country)

    fig = px.bar(europe_data, x='country_full', y='positivity_score', title='Positivity Scores in EU Countries')
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json





# Route to get most popular genres in EU


# Define genre mapping
genre_mapping = {
    'Tommy Richman': 'Pop',
    'Sabrina Carpenter': 'Pop',
    'Kendrick Lamar': 'Hip Hop',
    'Billie Eilish': 'Alternative/Indie',
    'Post Malone, Morgan Wallen': 'Hip Hop/Country',
    'FloyyMenor, Cris Mj': 'Latin/Reggaeton',
    'Artemas': 'Indie',
    'Hozier': 'Folk/Alternative',
    'Shaboozey': 'Hip Hop',
    'Benson Boone': 'Pop',
    'Taylor Swift, Post Malone': 'Pop/Hip Hop',
    'Ariana Grande': 'Pop',
    'Djo': 'Indie',
    'Teddy Swims': 'Soul/R&B',
    'Rvssian, Rauw Alejandro, Ayra Starr': 'Latin/Reggaeton',
    'Taylor Swift': 'Pop',
    'The Weeknd, JENNIE, Lily-Rose Depp': 'Pop/K-Pop',
    'Future, Metro Boomin, Kendrick Lamar': 'Hip Hop',
    'SZA': 'R&B/Soul',
    'Harry Styles': 'Pop'
}


# Map genres to the data
spotify_data['genre'] = spotify_data['artists'].map(genre_mapping)

# Filter data for Europe in 2023-2024
europe_data = spotify_data[
    ((spotify_data['snapshot_date'].str.startswith('2012')) | 
     (spotify_data['snapshot_date'].str.startswith('2024'))) &
    (spotify_data['country'].isin([
        'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU',
        'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
    ]))
].copy()

# Replace ISO codes with full country names
iso_to_country = {country.alpha_2: country.name for country in pycountry.countries}
europe_data['country_full'] = europe_data['country'].map(iso_to_country)

# Aggregate data to find the most popular genre for each European country in 2023-2024
europe_genre_counts = europe_data.groupby(['country_full', 'genre']).size().reset_index(name='count')

# Determine the most popular genre in each country
idx = europe_genre_counts.groupby(['country_full'])['count'].idxmax()
most_popular_genre_per_country = europe_genre_counts.loc[idx]

# Convert the genre data to a dictionary format
genre_data = {}
for index, row in most_popular_genre_per_country.iterrows():
    country = row['country_full']
    genre = row['genre']
    genre_data[country] = {'genre': genre, 'count': row['count']}

# Save the genre data to a JSON file
with open('genre_data.json', 'w') as json_file:
    json.dump(genre_data, json_file, indent=4)

# @app.route('/api/genres')
# def get_genres():
#     europe_data = spotify_data[
#         (spotify_data['snapshot_date'].str.startswith('2023')) |
#         (spotify_data['snapshot_date'].str.startswith('2024'))
#     ]
#     europe_data = europe_data[europe_data['country'].isin([
#         'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU',
#         'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
#     ])]

#     genre_counts = europe_data['genre'].value_counts().head(10)
#     fig = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values, title='Most Popular Genres in EU')
#     graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#     return graph_json


# Load genre data
with open('genre_data.json', 'r') as f:
    genre_data = json.load(f)

# Load Europe map data
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# europe = world[(world['continent'] == 'Europe')]

shapefile_path = './map/ne_110m_admin_0_countries.shp'

# Load the shapefile using GeoPandas
world = gpd.read_file(shapefile_path)

# List of European Union countries
eu_countries = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland',
    'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
]

# Filter for EU countries
europe = world[world['SOVEREIGNT'].isin(eu_countries)]


# Function to get genre from genre_data
def get_genre(country):
    return genre_data.get(country, {}).get('genre', None)


# Add genre information to the GeoDataFrame
europe['genre'] = europe['SOVEREIGNT'].apply(get_genre)
europe = europe[~europe['genre'].isnull()]


@app.route('/api/genres', methods=['GET'])
def get_genres():
    return jsonify(genre_data)

@app.route('/api/genres-map')
def get_genres_map():
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=europe['SOVEREIGNT'],
        z=europe['genre'].astype('category').cat.codes,
        text=europe['genre'],
        locationmode='country names',
        colorscale='Viridis',
        colorbar_title="Genre",
        hoverinfo='location+text'
    ))

    fig.update_layout(
        title_text='Most Popular Music Genre in Europe',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )

    return json.dumps(fig, cls=PlotlyJSONEncoder)

@app.route('/api/positivitychart')
def positivity_data():
    data = {
        "countries": ranked_countries['country_full'].tolist(),
        "scores": ranked_countries['positivity_score'].tolist()
    }
    return jsonify(data)


# Model performance data (should be computed as shown above)
# Assuming europe_data and features are defined as in the initial code
# Prepare the data for machine learning
X = europe_data[features]
y = europe_data['country']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    # 'SVM': SVC(kernel='linear', random_state=42),
    # 'Logistic Regression': LogisticRegression(random_state=42)
}

# Evaluate models
performance_data = []

for name, model in models.items():
    print(f"\nTraining {name} model...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Collect overall performance metrics
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        # Add to performance data
        performance_data.append({
            'model': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
    except Exception as e:
        print(f"{name} encountered an error: {e}")

# Save performance data to a JSON file
with open('performance_data.json', 'w') as f:
    json.dump(performance_data, f)


@app.route('/api/model-performance')
def get_model_performance():
    return jsonify(performance_data)


def create_top_10_playlist(country_name, model, data):
    country_songs = data[data['country_full'] == country_name]
    if not country_songs.empty:
        features_data = country_songs[features]
        predictions = model.predict(features_data)
        country_playlist = country_songs.copy()
        country_playlist['predicted_country'] = predictions
        top_10_playlist = country_playlist.sort_values(by='positivity_score', ascending=False).drop_duplicates(subset=['name']).head(10)
        return top_10_playlist[['name', 'artists', 'positivity_score']]
    else:
        return pd.DataFrame()

# Example: Create playlists and save to JSON
all_playlists = {}


for country in europe_data['country_full'].unique():
    playlist = create_top_10_playlist(country, models['Random Forest'], europe_data)
    if not playlist.empty:
        all_playlists[country] = playlist.to_dict(orient='records')

# Save playlists to JSON file
with open('playlists.json', 'w') as f:
    json.dump(all_playlists, f)


# Load playlist data
with open('playlists.json', 'r') as f:
    playlists_data = json.load(f)


@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    country = request.args.get('country')
    if country in playlists_data:
        return jsonify(playlists_data[country])
    else:
        return jsonify([]), 404


@app.route('/api/all-playlists', methods=['GET'])
def get_all_playlists():
    return jsonify(playlists_data)



if __name__ == '__main__':
    app.run(debug=True, port=5600)
