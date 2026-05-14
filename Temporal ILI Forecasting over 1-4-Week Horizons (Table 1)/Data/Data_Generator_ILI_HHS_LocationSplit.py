import pandas as pd
import numpy as np
import argparse
import datetime
from sklearn.preprocessing import MinMaxScaler
from Utils import drop_last_n_samples, splitter

# Function to get the start date of a given year and week
def get_date_from_year_week(year, week):
    try:
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w").date()
    except ValueError:
        # Week 53 issue: Assume last Monday of the year if invalid week
        last_day_of_year = datetime.date(year, 12, 31)
        return last_day_of_year - datetime.timedelta(days=last_day_of_year.weekday())
    
    
def prepare_ILI_data(path_to_data, cross_validation_part):
    # Load data
    data = pd.read_csv(path_to_data)
    print(data)
    data = data[["YEAR", "WEEK", "REGION", "% WEIGHTED ILI",'AGE 0-4',  'AGE 25-49',  'AGE 25-64',  'AGE 5-24',  'AGE 50-64',  'AGE 65',  'ILITOTAL']]
    data = data.fillna(0)
    data.sort_values(by=['REGION', "YEAR", "WEEK"], inplace=True)
    data = data.reset_index(drop=True)

    # Split data to remove noisy part
    _, data = splitter(data, 'REGION', 1000)
    data = data.reset_index(drop=True)

    # Add Indicator instead of date
    data["Indicator"] = data.groupby("REGION").cumcount() + 1

    # Assuming your DataFrame is named ili_df
    data["DATE"] = data.apply(lambda row: get_date_from_year_week(row["YEAR"], row["WEEK"]), axis=1)

    # Prepare the final DataFrame
    df = data[["Indicator", "REGION", "% WEIGHTED ILI", "WEEK",'AGE 0-4',  'AGE 25-49',  'AGE 25-64',  'AGE 5-24',  'AGE 50-64',  'AGE 65',  'ILITOTAL']]
    df.columns = ["ds", "unique_id", "y", "WEEK",'AGE 0-4',  'AGE 25-49',  'AGE 25-64',  'AGE 5-24',  'AGE 50-64',  'AGE 65',  'ILITOTAL']
    df.loc[:, 'y'] = df['y'].astype(float)

    number_of_time_series = len(df['unique_id'].unique())
    length_time_series = int(len(df) / len(df['unique_id'].unique()))
    forecasting_horizon= 4

    # Identify unique time series
    unique_ids = df['unique_id'].unique()

    if cross_validation_part == 1:
        train_ids = ['Region 3', 'Region 4', 'Region 5', 'Region 6', 'Region 7', 'Region 8','Region 9', 'Region 10']
        test_ids = ['Region 1', 'Region 2']
    if cross_validation_part == 2:
        train_ids = ['Region 1', 'Region 2', 'Region 5', 'Region 6', 'Region 7', 'Region 8','Region 9', 'Region 10']
        test_ids = ['Region 3', 'Region 4']
    if cross_validation_part == 3:
        train_ids = ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 7', 'Region 8','Region 9', 'Region 10']
        test_ids = ['Region 5', 'Region 6']
    if cross_validation_part == 4:
        train_ids = ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5', 'Region 6', 'Region 9', 'Region 10']
        test_ids = ['Region 7', 'Region 8']        
    if cross_validation_part == 5:
        train_ids = ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5', 'Region 6', 'Region 7', 'Region 8']
        test_ids = ['Region 9', 'Region 10']        
          
    # Create train and test DataFrames
    train_df = df[df['unique_id'].isin(train_ids)]
    test_df = df[df['unique_id'].isin(test_ids)]


    return train_df, test_df, df, data, number_of_time_series, length_time_series, forecasting_horizon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for TimeLLM model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--cross_validation_part", type=int, default=1, help="Part of data as validation.")
    args = parser.parse_args()

    train_df, test_df, df, data, number_of_time_series, length_time_series, forecasting_horizon = prepare_data(args.data_path, args.cross_validation_part)
    print("Data preparation complete.")
    
    
    
    



