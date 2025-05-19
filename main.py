from time import sleep
import pandas as pd
import logging

from classifier.data_loader import load_file
from classifier.ad_analyzer import AdAnalyzer

logger = logging.getLogger(__name__)

# Path for raw data, as downloaded:
raw_path = 'data/raw/ads_with_transcripts_and_ideal_points.csv'
# Path for data with ony the relevant columns, before classification:
processed_path = 'data/processed/ads_with_transcripts.csv'
# Path for data with the classification results:
classified_path = 'data/processed/classification_results.csv'

relevant_cols = ['_id', 'bylines', 'ad_creative_bodies', 'ad_creative_link_titles', 'page_name',
                 'transcript_translated']


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Starting process")


def update_with_previous_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if previous results exist to avoid running the same ads again.
    :param df:
    :return: DataFrame with only the ads that need to be classified.
    """

    try:
        df_old = pd.read_csv(classified_path, index_col="id")
    except FileNotFoundError:
        logger.warning(f"Previous results file not found at {classified_path}. Starting fresh.")
        return df

    ids_already_classified = df.index.intersection(df_old.index)

    # Return only the rows that have not been classified yet
    df_to_analyze = df.loc[~df.index.isin(ids_already_classified)]

    logger.info(f"Found {len(df_to_analyze)} ads that need to be classified.")

    return df_to_analyze


def merge_with_old_results(df_new: pd.DataFrame, classified_path: str) -> pd.DataFrame:
    """
    Loads previous classification results (if available) and merges them with the new results.
    Keeps the latest entry for any duplicate IDs.

    :param df_new: Newly classified ads (DataFrame with "_id" as index).
    :param classified_path: Path to the previously saved classification results.
    :return: Combined DataFrame with old + new results.
    """
    try:
        df_old = pd.read_csv(classified_path, index_col="id")
        logger.info(f"Loaded {len(df_old)} previously classified ads.")
    except FileNotFoundError:
        logger.warning(f"No previous results at {classified_path}. Starting fresh.")
        return df_new

    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
    logger.info(f"Combined total: {len(df_combined)} classified ads.")

    return df_combined


def iterate_over_df(df: pd.DataFrame, analyzer: AdAnalyzer) -> pd.DataFrame:
    """
    Iterate over the ads and classify each one using the AdAnalyzer.
    :param df:
    :param analyzer: AdAnalyzer instance
    :return: Results DF only with the classification status
    """
    logging.info("Starting classification process.")
    results = []

    for i, row in enumerate(df.itertuples()):
        try:
            ad_result = analyzer.analyze(**row._asdict())
            results.append(ad_result)
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1} ads out of {len(df)}")
                df_tmp = pd.DataFrame(results).set_index("id")
                df_tmp.to_csv(f"data/processed/classification_results_tmp.csv")
        except Exception as e:
            logger.warning(f"Error processing ad {row[0]}: {e}")
            results.append({
                "id": row[0],
                "classification": "Error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        # max calls: 1000/min
        sleep(0.1)

    df_results = pd.DataFrame(results).set_index("id")
    logging.info("Finished classification process.")
    return df_results


def run_classification(model: str, temperature: float = 0, subset: int = 0) -> pd.DataFrame:
    """
    Main function to run the classification process for ads.
    :param model:
    :param temperature:
    :param subset: specify a subset of ads to classify, 0 for all
    :return:
    """
    df = load_file(relevant_cols=relevant_cols, raw_path=raw_path, processed_path=processed_path)
    analyzer = AdAnalyzer(model=model, temperature=temperature)

    if subset > 0:
        df = df.sample(subset)

    df = update_with_previous_results(df)

    df_results = iterate_over_df(df, analyzer)

    df_combined = merge_with_old_results(df_results, classified_path)
    df_combined.to_csv(classified_path)

    return df_combined


def get_classification_for_sample(sample_path: str = "data/processed/sample_labeled.csv"):
    """
    Run to get the model's classification for the manually labeled sample
    :param sample_path:
    :return:
    """
    model = "llama-3.1-8b-instant"
    temperature = 0

    # Load the data and instantiate the analyzer
    df = load_file(relevant_cols=relevant_cols, raw_path=raw_path, processed_path=processed_path)
    analyzer = AdAnalyzer(model=model, temperature=temperature)

    # Select the ids from the labeled sample
    sample_df = pd.read_csv(sample_path)
    df_to_analyze = pd.merge(sample_df[['_id']], df, on="_id", how="left")

    # Process the sample ads through the conventional analyzer
    df_to_analyze = df_to_analyze.set_index("_id")
    df_results = iterate_over_df(df_to_analyze, analyzer)
    df_results.to_csv("data/processed/sample_labeled_classified_3.csv")


def main():
    """
    Runs the classification process for all ads and saves the results.
    To avoid running all the ads, use the subset parameter.
    If you want to run the classification for the sample, uncomment the get_classification_for_sample function.

    :return: None
    """
    # df_combined = get_classification_for_sample()

    df_combined = run_classification(model="llama-3.1-8b-instant", temperature=0, subset=0)
    df_combined.to_csv(classified_path)


if __name__ == '__main__':
    setup_logger()  # Logging configuration
    main()
