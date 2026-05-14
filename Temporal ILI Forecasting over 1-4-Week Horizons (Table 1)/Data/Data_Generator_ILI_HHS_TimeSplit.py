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
    
    
def prepare_ILI_data(path_to_data, test_size):
    # Load data
    data = pd.read_csv(path_to_data)
    data = data[["YEAR", "WEEK", "REGION", "% WEIGHTED ILI"]]
    data = data.fillna(data.mean(numeric_only=True))
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
    df = data[["Indicator", "REGION", "% WEIGHTED ILI"]]
    df.columns = ["ds", "unique_id", "y"]
    df.loc[:, 'y'] = df['y'].astype(float)

    # Split into train and test
    number_of_time_series = len(df['unique_id'].unique())
    length_time_series = int(len(df) / number_of_time_series)
    forecasting_horizon = 4
    number_of_samples_in_test = int(length_time_series * test_size) - (int(length_time_series * test_size) % forecasting_horizon)
    number_of_samples_in_train = length_time_series - number_of_samples_in_test

    train_df, test_df = splitter(df, 'unique_id', number_of_samples_in_test)

    return train_df, test_df, df, data, number_of_time_series, length_time_series, forecasting_horizon, number_of_samples_in_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for TimeLLM model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    args = parser.parse_args()

    train_df, test_df, df, data, number_of_time_series, length_time_series, forecasting_horizon, number_of_samples_in_test = prepare_data(args.data_path, args.test_size)
    print("Data preparation complete.")
    
    
    
    



