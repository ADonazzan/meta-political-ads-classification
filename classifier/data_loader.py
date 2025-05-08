import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

raw_path = '../data/raw/ads_with_transcripts_and_ideal_points.csv'
processed_path = '../data/processed/ads_with_transcripts.csv'


def load_file(relevant_cols, raw_path=raw_path, processed_path=processed_path) -> pd.DataFrame:
    """
    Load and preprocess the ad data for classification.
    Loads from cached 'processed' version if available and valid.
    :return: DataFrame with _id as index and cleaned relevant columns.
    """

    expected_cols = relevant_cols[1:]  # Skip _id for column check

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, index_col=0)
        if all(col in df.columns for col in expected_cols):
            logger.info("Loaded processed file successfully.")
            return df
        else:
            logger.warning("Processed file columns don't match expected. Reprocessing.")
            os.remove(processed_path)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    df = pd.read_csv(raw_path, index_col='_id', usecols=relevant_cols)
    df.to_csv(processed_path)
    logger.info("Processed raw file and saved cleaned version.")

    return df


if __name__ == '__main__':
    # Load the file
    df = load_file()
