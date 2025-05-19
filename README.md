# Identification of Advertisements related to Presidential Elections

The `main.py` file contains the code to classify advertisements.
It assumes you have the ads in the `data/raw` folder, with the id column named `"_id"`.

**Before running the code**, add the `GROQ_API_KEY` field to the .env file.

### Processing speed
This code processes roughly 200 ads in one minute, 95k ads in 8 hours.

### Token Consumption
Each request takes up 312 tokens on average, for 95k ads, that's 29.6M tokens.
using Llama 3.1 8B for 20M / $1, it's $1.5

# Results
Results from llama 3.1 8b instant, compared to the manual classification of 192 ads

    'True Positives': 22,
    'True Negatives': 164,
    'False Positives': 4,
    'False Negatives': 3,
    'Accuracy': 0.964,
    'Precision': 0.846,
    'Recall': 0.88,
    'F1 Score': 0.863

Results from llama 3.3 70b versatile, compared to manual classification

    'True Positives': 22,
    'True Negatives': 186,
    'False Positives': 8,
    'False Negatives': 2,
    'Accuracy': 0.954,
    'Precision': 0.733,
    'Recall': 0.917,
    'F1 Score': 0.815

Results from llama 3.3 70b versatile, compared to llama 3.1 8B (sample of 2000)

    'True Positives': 237,
    'True Negatives': 1645,
    'False Positives': 45,
    'False Negatives': 66,
    'Accuracy': 0.944,
    'Precision': 0.84,
    'Recall': 0.782,
    'F1 Score': 0.81

## Most common errors
Some from HARRIS VICTORY FUND (7/127) get labeled as non-presidential.


> Did you see this? Millennial and Gen Z voters in North Carolina could decide this year's election for President! 
> Not just that, but young voters have the power to pick our Governor, along with a seat on the NC Supreme Court, other 
> statewide contests, and more.

Labeled as presidential, but the model classified it as non-presidential.

> New Hoodie Goes Viral After Trump Wins...

Labeled as non-presidential, but the model classifies it as presidential.

> Step into the action and bet on the U.S. Election with Kalshi

Labeled as presidential, but the model classifies it as non-presidential.

> Ready to win big? Don't make the mistake of skipping early research! Chart your course to victory by leveraging polls and text messages to understand what drives your voters. Get a FREE voter profile of your district at https://win.bestcandidate.us/2024win

Labeled as non-presidential, but the model classifies it as presidential.