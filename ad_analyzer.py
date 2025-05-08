import os
from dotenv import load_dotenv
import logging

from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


class AdAnalyzer:
    def __init__(self, model: str, temperature: float = 0):
        self.model = model
        self.temperature = temperature
        api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=api_key)

    def _send_request(self, prompt: str) -> dict:
        chat = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict classifier. Only respond with 'Presidential', 'Other', or 'Unsure'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return chat.to_dict()

    def _create_prompt(self, **kwargs) -> str:
        prompt = (
            """Below, please find the transcript and/or the post associated with a political advertisement video from social media.\n
            Please reply whether it is about the US *Presidential* Election 2024 (Trump vs. Harris) or any other election
            (that may or may not have taken place on the same day).\n
            Reply ONLY USING \"Presidential\", \"Other\", or \"Unsure\" and NOTHING ELSE.
            """
        )

        # Add relevant ad fields with headers if available
        field_labels = {
            'ad_creative_bodies': "Ad Text",
            'ad_creative_link_titles': "Link Title",
            'transcript_translated': "Transcript",
            'bylines': "Bylines",
            'ad_delivery_start_time': "Start Time",
            'ad_delivery_stop_time': "Stop Time"
        }

        for key, label in field_labels.items():
            value = kwargs.get(key)
            if value:
                prompt += f"{label}:\n{value}\n\n"

        return prompt.strip()

    def _parse_response(self, response: dict) -> dict:
        try:
            content = response["choices"][0]["message"]["content"].strip()
            classification = content if content in {"Presidential", "Other", "Unsure"} else "InvalidResponse"

            usage = response.get("usage", {})
            token_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            return {
                "classification": classification,
                "tokens": token_info
            }

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to parse response: {e} | Raw response: {response}")
            return {
                "classification": "Error",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    def analyze_ad(self, **kwargs) -> dict:
        try:
            prompt = self._create_prompt(**kwargs)
            response = self._send_request(prompt)
            result = self._parse_response(response)
            return {
                "id": kwargs.get("_id", None),
                "classification": result["classification"],
                "tokens": result["tokens"]
            }
        except Exception as e:
            logger.warning(f"Error analyzing ad {kwargs.get('_id', 'UNKNOWN')}: {e}")
            return {
                "id": kwargs.get("_id", None),
                "classification": "Error",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }


if __name__ == '__main__':
    model = 'llama-3.1-8b-instant'
    temperature = 0

    ad = {
        "_id": "1946402405848135",
        "ad_creative_bodies": (
            "We have entered the final stretch of this race, and we need to look at Donald Trump and remember -- I mean really remember -- who he is.\n\n"
            "Trump is the guy who prefers to run on problems instead of finding solutions.\n\n"
            "Trump is the guy who spends his time name-calling and demeaning his fellow Americans -- including our veterans.\n\n"
            "Trump is the guy who said that women should be punished for having abortions.\n\n"
            "That is who Donald Trump and his allies are. Now, we must remember who we are and fight for the country we so love. Donate right away."
        ),
        "ad_creative_link_titles": ["Now or never"],
        "ad_delivery_start_time": "2024-10-28",
        "ad_delivery_stop_time": "2024-10-29",
        "bylines": "HARRIS VICTORY FUND",
        "transcript_translated": "you"
    }

    ad_analyzer = AdAnalyzer(model=model, temperature=temperature)
    result = ad_analyzer.analyze_ad(**ad)
    print(result)
