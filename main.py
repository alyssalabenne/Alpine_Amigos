import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from simplify import simplify
from preprocess_data import preprocess_data
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')

app = Flask(__name__)


def preprocess_all_data_together(user_input):
    # Read the ski resort data
    ski_data = pd.read_csv('data/simplified_ski_resorts.csv')
    user_input = process_user_input(user_input)
    user_input = pd.DataFrame(user_input, index=[0])

    # Concatenate resort data and user data
    combined_data = pd.concat([ski_data, user_input], ignore_index=True)

    # Save combined data to a new CSV file
    combined_data.to_csv('data/combined_data.csv', index=False)

    preprocess_data()


def preprocess_user_input(user_preferences):
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv')

    # Convert dictionary to DataFrame
    user_preferences_df = pd.DataFrame(user_preferences, index=[0])

    # Define categorical and numerical features
    categorical_features = ['country', 'continent', 'snowparks']
    numerical_features = ['price', 'total_slopes', 'beginner_slopes', 'intermediate_slopes', 'difficult_slopes']

    encoded_categorical_features = pd.get_dummies(user_preferences_df[categorical_features])

    # Combine encoded categorical features and numerical features
    preprocessed_data = pd.concat([encoded_categorical_features, user_preferences_df[numerical_features]], axis=1)

    # Add the 'Resort' column
    preprocessed_data['resort'] = user_preferences_df['resort']

    return preprocessed_data


# Function to process user input
def process_user_input(user_input):
    # Add resort id
    user_input['id'] = 500
    # Add resort name if available
    user_input['resort'] = "Desired Resort"
    # Add resort country
    if user_input['country'] == '':
        user_input['country'] = 'Any Country'
    # Convert total slopes category into numerical values
    if user_input['total_slopes'] == 'Small':
        user_input['total_slopes'] = 40
    elif user_input['total_slopes'] == 'Medium':
        user_input['total_slopes'] = 100
    elif user_input['total_slopes'] == 'Large':
        user_input['total_slopes'] = 200

    # Convert slope difficulty category into numerical values
    if user_input['slope_difficulty'] == 'Beginner':
        user_input['beginner_slopes'] = 20
        user_input['intermediate_slopes'] = 0
        user_input['difficult_slopes'] = 0
    elif user_input['slope_difficulty'] == 'Intermediate':
        user_input['beginner_slopes'] = 0
        user_input['intermediate_slopes'] = 20
        user_input['difficult_slopes'] = 0
    elif user_input['slope_difficulty'] == 'Difficult':
        user_input['beginner_slopes'] = 0
        user_input['difficult_slopes'] = 20
        user_input['intermediate_slopes'] = 0

    # Drop slope_difficulty key
    user_input.pop('slope_difficulty', None)

    return user_input


# Function to load preprocessed data and perform similarity calculation
def run_similarity_calculation():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv')

    # Define features for similarity calculation
    features = ['continent_Asia', 'continent_Europe', 'continent_North America', 'continent_Oceania',
                'continent_South America', 'price', 'total_slopes', 'beginner_slopes', 'intermediate_slopes',
                'difficult_slopes']

    # Standardize numerical features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(df[features])

    # Convert similarity matrix to DataFrame for better visualization
    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

    # Example: Find similar ski resorts to a target resort
    target_resort_index = 499  # Change this index to select a different target resort
    similar_resorts = similarity_df[target_resort_index].sort_values(ascending=False)

    # Normalize similarity scores to [0, 1]
    min_score = similar_resorts.min()
    max_score = similar_resorts.max()
    similar_resorts = (similar_resorts - min_score) / (max_score - min_score)

    return similar_resorts


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/recommendations/', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        country = request.form.get('country')
        continent = request.form.get('continent')
        price = request.form.get('price')
        total_slopes = request.form.get('total_slopes')
        slope_difficulty = request.form.get('slope_difficulty')
        snowparks = request.form.get('snowparks')

        # Create user input dictionary
        user_input = {
            'country': country,
            'continent': continent,
            'price': price,
            'slope_difficulty': slope_difficulty,
            'total_slopes': total_slopes,
            'snowparks': snowparks
        }

        preprocess_all_data_together(user_input)

        simplify()

        # Get similar resorts and their similarity scores
        similar_resorts_data = run_similarity_calculation()

        # Exclude user input resort from the list
        user_input_resort_index = 499  # Assuming user input resort index
        similar_resorts_data = similar_resorts_data.drop(user_input_resort_index, errors='ignore')

        # Filter out only the indexes of similar resorts
        similar_resort_indexes = similar_resorts_data.index

        # Read the resort data from the CSV file
        resort_data = pd.read_csv('data/combined_data.csv')

        # Filter the resort data based on similar resort indexes
        similar_resorts = resort_data.loc[similar_resort_indexes]

        # Add similarity scores to the resort data
        similar_resorts['similarity_score'] = similar_resorts_data.values

        return render_template('recommendations.html', similar_resorts=similar_resorts)

    return render_template('recommendations.html')


@app.route('/pie_chart/')
def pie_chart():
    # Get similarity scores
    similarity_scores = run_similarity_calculation()

    # Define bins and labels
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Very Dissimilar', 'Dissimilar', 'Moderate', 'Similar', 'Very Similar']
    colors = ['#c9c3c2', '#465E6D', '#7D95A4', '#EEECE8', '#f7f7f7']
    explode = (0, 0, 0, 0, 0.3)

    # Create histogram to count scores in each bin
    hist, _ = np.histogram(similarity_scores, bins=bins)

    # Plot the pie chart
    plt.pie(hist, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%')
    plt.title('Similarity Score Distribution')

    # Save the plot as a PNG file

    # Define the file path
    file_path = 'static/pie_chart.png'

    # Check if the file exists
    if os.path.exists(file_path):
        # If it exists, delete the file
        os.remove(file_path)

    plt.show()
    # Save the plot as a PNG file
    plt.savefig(file_path)

    return render_template('pie_chart.html')


if __name__ == '__main__':
    app.run(debug=True)
