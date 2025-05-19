import pandas as pd
import ast


def select_sample():
    """
    Select a sample of ads, 100 with and 100 without transcripts for manual labeling.
    :return:
    """
    raw_path = 'data/raw/ads_with_transcripts_and_ideal_points.csv'

    df = pd.read_csv(raw_path, low_memory=False)

    sample = df[df.transcript_translated.notna()].sample(100)
    sample2 = df[df.transcript_translated.isna()].sample(100)

    sample = pd.concat([sample, sample2])

    sample.to_csv('data/processed/sample_labeled.csv', index=False)


def merge_labeled_classified():
    """
    Merge the manually labeled sample with the sample classified by the model to check for errors.
    :return: None
    """
    df_labeled = pd.read_csv('data/processed/sample_labeled.csv')
    df_class = pd.read_csv('data/processed/sample_labeled_classified_3.csv')

    df_merged = pd.merge(df_labeled, df_class, left_on='_id', right_on='id', how='left')
    df_merged['class_int'] = [1 if x == 'Presidential' else 0 for x in df_merged['classification']]
    df_merged['error'] = df_merged['is_presidential'] != df_merged['class_int']

    df_merged.to_csv('data/processed/sample_labeled_merged_3.csv', index=False)