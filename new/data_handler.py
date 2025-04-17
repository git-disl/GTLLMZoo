# data_handler.py
import pandas as pd
import os
import logging
import numpy as np # Import numpy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_FILE_PATH = os.path.join('data', 'merged_leaderboards.csv')
KEY_COLUMN = 'model_std' # The column that determines if subsequent columns should be NA

def load_leaderboard_data():
    """
    Loads the leaderboard data from the CSV file and processes it.

    Returns:
        pandas.DataFrame: The processed leaderboard data, or None if an error occurs.
    """
    if not os.path.exists(DATA_FILE_PATH):
        logging.error(f"Data file not found at: {DATA_FILE_PATH}")
        return None
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        logging.info(f"Successfully loaded data from {DATA_FILE_PATH}. Shape: {df.shape}")

        # Find the index of the key column
        try:
            key_col_index = df.columns.get_loc(KEY_COLUMN)
            # Ensure index calculation doesn't go out of bounds
            if key_col_index + 1 < len(df.columns):
                 cols_to_nan = df.columns[key_col_index + 1:]
                 logging.info(f"Columns to potentially set NA (if '{KEY_COLUMN}' is empty): {list(cols_to_nan)}")
            else:
                 cols_to_nan = pd.Index([]) # Empty index if key column is the last one
                 logging.info(f"Key column '{KEY_COLUMN}' is the last column, no subsequent columns to modify.")

        except KeyError:
            logging.warning(f"Key column '{KEY_COLUMN}' not found in the CSV. Skipping NA replacement logic.")
            cols_to_nan = pd.Index([]) # Empty index

        # Identify rows where the key column is null or empty string
        # pd.isna() catches NaN, None. Also check for empty strings explicitly.
        # Ensure KEY_COLUMN exists before attempting to access it
        if KEY_COLUMN in df.columns:
            is_na_condition = pd.isna(df[KEY_COLUMN]) | (df[KEY_COLUMN] == '')
        else:
            # If KEY_COLUMN doesn't exist, the condition is false for all rows
            is_na_condition = pd.Series([False] * len(df), index=df.index)


        # Set subsequent columns to "NA" where the condition is met
        # Check if there are columns to modify AND if any rows meet the condition
        if not cols_to_nan.empty and is_na_condition.any():
            logging.info(f"Preparing to apply 'NA' to {is_na_condition.sum()} rows for columns: {list(cols_to_nan)}")

            # Convert relevant columns to object type FIRST to avoid dtype warnings/errors
            for col in cols_to_nan:
                 # Check if column exists and if its dtype is not already object
                 if col in df.columns and df[col].dtype != 'object':
                     logging.info(f"Changing dtype of column '{col}' from {df[col].dtype} to object to accommodate 'NA' string.")
                     try:
                         df[col] = df[col].astype(object)
                     except Exception as e:
                         logging.error(f"Could not convert column '{col}' to object type: {e}")

            # Now assign "NA" using .loc
            # Using numpy's nan might be better if downstream processes handle it,
            # but using the string "NA" as requested.
            try:
                 # Use df.loc[row_indexer, col_indexer] = value
                 df.loc[is_na_condition, cols_to_nan] = "NA"
                 logging.info(f"Successfully applied 'NA' string.")
            except Exception as e:
                 # Log potential errors during assignment
                 logging.error(f"Error assigning 'NA' value using .loc: {e}", exc_info=True)


        # Optional: Fill remaining NaNs in the *entire* dataframe if desired
        # Consider using pd.NA instead of "NA" string if appropriate
        # df.fillna(pd.NA, inplace=True) # Modern pandas NA
        # df.fillna("NA", inplace=True) # Using string "NA"
        # logging.info("Filled any remaining NaN values across the dataframe with 'NA'")

        return df

    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {DATA_FILE_PATH}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Error: Data file {DATA_FILE_PATH} is empty.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading or processing the CSV: {e}", exc_info=True) # Add exc_info for more detail
        return None

# Keep the testing block if needed
if __name__ == '__main__':
    print("Testing data handler...")
    data = load_leaderboard_data()
    if data is not None:
        print("\nData loaded successfully:")
        print(data.head())
        print("\nInfo after potential NA replacement:")
        # Check a column that might have been affected
        if KEY_COLUMN in data.columns:
             key_col_idx_test = data.columns.get_loc(KEY_COLUMN)
             if key_col_idx_test + 1 < len(data.columns):
                 affected_col_name = data.columns[key_col_idx_test+1]
                 print(f"\nValue counts for '{affected_col_name}' (example affected column):")
                 print(data[affected_col_name].value_counts(dropna=False))
             else:
                 print(f"No columns after '{KEY_COLUMN}' to check.")

        else:
             print(f"Key column '{KEY_COLUMN}' not found.")

    else:
        print("\nFailed to load data.")