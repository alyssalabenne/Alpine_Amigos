import pandas as pd


def simplify():
    # read_csv function which is used to read the required CSV file
    data = pd.read_csv('data/ski_resorts.csv')

    # drop function which is used in removing or deleting rows or columns from the CSV files
    data.drop('Latitude', inplace=True, axis=1)
    data.drop('Longitude', inplace=True, axis=1)
    data.drop('Highest point', inplace=True, axis=1)
    data.drop('Lowest point', inplace=True, axis=1)
    data.drop('Snow cannons', inplace=True, axis=1)
    data.drop('Longest run', inplace=True, axis=1)
    data.drop('Surface lifts', inplace=True, axis=1)
    data.drop('Chair lifts', inplace=True, axis=1)
    data.drop('Gondola lifts', inplace=True, axis=1)
    data.drop('Lift capacity', inplace=True, axis=1)
    data.drop('Total lifts', inplace=True, axis=1)
    data.drop('Child friendly', inplace=True, axis=1)
    data.drop('Nightskiing', inplace=True, axis=1)
    data.drop('Summer skiing', inplace=True, axis=1)
    data.drop('Season', inplace=True, axis=1)

    # Save combined data to a new CSV file
    data.to_csv('data/simplified_ski_resorts.csv', index=False)