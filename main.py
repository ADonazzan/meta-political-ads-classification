from classifier.data_loader import load_file
from classifier.ad_analyzer import AdAnalyzer


def main():
    relevant_cols = [
        '_id', 'ad_creative_bodies', 'ad_creative_link_titles',
        'ad_delivery_start_time', 'ad_delivery_stop_time', 'bylines',
        'transcript_translated'
    ]

    # Load the ad data
    df = load_file(relevant_cols)

