import pandas as pd
import ast


def select_sample(sample_size=100):
    """
    Select a sample of ads, 100 with and 100 without transcripts for manual labeling.
    :return:
    """
    raw_path = 'data/raw/ads_with_transcripts_and_ideal_points.csv'

    df = pd.read_csv(raw_path, low_memory=False)

    sample = df[df.transcript_translated.notna()].sample(sample_size)
    sample2 = df[df.transcript_translated.isna()].sample(sample_size)

    sample = pd.concat([sample, sample2])

    sample.to_csv('data/processed/sample_labeled.csv', index=False)


def find_ad(id_to_find) -> pd.DataFrame:
    """
    Return ad with the given id.
    :param id_to_find:
    :return: DataFrame with the ad.
    """
    df = pd.read_csv('data/raw/ads_with_transcripts_and_ideal_points.csv', low_memory=False)
    return df[df['_id'] == int(id_to_find)]


def compute_metrics(df_merged):
    """
    Compute classification metrics for the merged DataFrame.
    :param df_merged:
    :return:
    """
    df_merged['is_presidential'] = df_merged['is_presidential'].astype(bool)
    df_merged['classification'] = df_merged['classification'].astype(bool)

    TP = ((df_merged['is_presidential'] == True) & (df_merged['classification'] == True)).sum()
    TN = ((df_merged['is_presidential'] == False) & (df_merged['classification'] == False)).sum()
    FP = ((df_merged['is_presidential'] == False) & (df_merged['classification'] == True)).sum()
    FN = ((df_merged['is_presidential'] == True) & (df_merged['classification'] == False)).sum()

    metrics = {
        'True Positives': int(TP),
        'True Negatives': int(TN),
        'False Positives': int(FP),
        'False Negatives': int(FN),
        'Accuracy': float(round((TP + TN) / len(df_merged), 3)),
        'Precision': float(round(TP / (TP + FP), 3)) if TP + FP > 0 else None,
        'Recall': float(round(TP / (TP + FN), 3)) if TP + FN > 0 else None,
        'F1 Score': float(round(2 * TP / (2 * TP + FP + FN), 3)) if TP + FP + FN > 0 else None
    }
    return metrics


def merge_labeled_classified():
    """
    Merge the manually labeled sample with the sample classified by the model to check for errors.
    """
    df_labeled = pd.read_csv('data/processed/sample_labeled.csv')
    df_class = pd.read_csv('data/processed/sample_labeled_classified.csv', usecols=['id', 'classification'])

    df_merged = pd.merge(df_labeled, df_class, left_on='_id', right_on='id', how='left')
    df_merged['classification'] = [1 if x == 'Presidential' else 0 for x in df_merged['classification']]
    df_merged['error'] = df_merged['is_presidential'] != df_merged['classification']

    df_merged.drop(columns=['id'], inplace=True)

    df_merged.to_csv('data/processed/sample_labeled_merged.csv', index=False)

    metrics = compute_metrics(df_merged)
    return metrics


def merge_original_classified():
    """
    Merge the original dataset with the classification output and save the result.
    """
    # Path for raw data, as downloaded:
    raw_path = 'data/raw/ads_with_transcripts_and_ideal_points.csv'
    # Path for data with the classification results:
    classified_path = 'data/processed/classification_results.csv'
    # Path for the merged output:
    merged_path = 'data/processed/ads_with_classification.csv'

    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_classified = pd.read_csv(classified_path, usecols=['id', 'classification'])

    # Check if all ids are present in the classified data
    if df_raw['_id'].nunique() != df_classified['id'].nunique():
        raise ValueError("Not all ids are present in the classified data.")

    df_classified["is_presidential"] = [1 if x == 'Presidential' else 0 for x in df_classified['classification']]
    df_classified.drop(columns=['classification'], inplace=True)

    df_merged = pd.merge(df_raw, df_classified, left_on='_id', right_on='id', how='left')
    df_merged.drop(columns=['id'], inplace=True)

    df_merged.to_csv(merged_path, index=False)


if __name__ == '__main__':
    # select_sample(1000)
    print(merge_labeled_classified())
    # merge_original_classified()
