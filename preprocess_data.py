import pandas as pd


def preprocess_data():
    # Read the ski resort data
    ski_data = pd.read_csv('data/combined_data.csv')

    # Define categorical and numerical features
    categorical_features = ['country', 'continent', 'snowparks']
    numerical_features = ['price', 'total_slopes', 'beginner_slopes', 'intermediate_slopes', 'difficult_slopes']

    # Perform data preprocessing
    # Encoding categorical variables
    encoded_categorical_features = pd.get_dummies(ski_data[categorical_features])

    # Combine encoded categorical features and numerical features
    preprocessed_data = pd.concat([encoded_categorical_features, ski_data[numerical_features]], axis=1)

    # Add the 'Resort' column
    preprocessed_data['resort'] = ski_data['resort']

    # Save preprocessed data to CSV
    preprocessed_data.to_csv('data/preprocessed_data.csv', index=False)
