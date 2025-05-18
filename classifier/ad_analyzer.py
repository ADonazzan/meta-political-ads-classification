import os
from dotenv import load_dotenv
import logging
from retry import retry
import re
import ast
import pandas as pd

from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


class AdAnalyzer:
    def __init__(self, model: str, temperature: float):
        self.base_prompt = (
            "Your task is to determine if the ad is **explicitly** about the 2024 US Presidential Election.\n"
            "Only classify the ad as:\n"
            '- "Presidential": if it directly mentions the 2024 Presidential Election, backs, attacks or mentions a candidate (Trump or Harris), mentions campaign slogans, calls to action or events/conventions.\n'
            '- "Other": political content unrelated to the 2024 US Presidential race.\n'
            '- "Unsure": if the ad lacks enough information to determine its context.\n'
            'Reply with only one of: "Presidential", "Other", or "Unsure". Do not explain your answer.\n\n'
        )
        self.model = model
        self.temperature = temperature
        api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=api_key)

    @retry(tries=3, delay=10, backoff=2, logger=logger)
    def _send_request(self, prompt: str) -> dict:
        chat = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict classifier of political ads. Only respond with 'Presidential', 'Other', or 'Unsure'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return chat.to_dict()

    def _clean_text(self, text: str) -> str:
        max_transcript_length = 400

        # Remove repeated filler phrases
        text = re.sub(r"\b(uh|um|you know|like|I mean|just|really|so|okay|literally|actually)\b", "", text,
                      flags=re.IGNORECASE)
        text = re.sub(r"\n", "", text)
        text = re.sub(r"\s+", " ", text)  # remove extra spaces

        # Extract hashtags from the cleaned text
        hashtags = list(dict.fromkeys(re.findall(r"#\w+", text)))  # deduplicated

        if len(text) > max_transcript_length:
            text = text[:max_transcript_length] + "..."

        # Append hashtags at the end
        text = f"{text} {' '.join(hashtags)}".strip()
        return text

    def _create_prompt(self, **kwargs) -> str:
        # Define field labels and desired order
        ordered_fields = [
            ('bylines', "Bylines"),
            ('ad_creative_link_titles', "Link Title"),
            ('ad_creative_bodies', "Ad Text"),
            ('page_name', "Page Name"),
            ('ad_delivery_stop_time', "Stop Time"),
            ('transcript_translated', "Transcript"),
        ]

        tmp_prompt = ''
        for key, label in ordered_fields:
            value = kwargs.get(key)
            if pd.isna(value):
                continue

            # Handle list-like values (e.g., ['a', 'b'] or '["a";"b"]')
            # If it's a string that looks like a list, parse it safely
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list) and parsed:
                        value = str(parsed[0])
                except Exception:
                    pass  # leave value as-is if parsing fails

            if key in {"ad_creative_bodies", "transcript_translated"}:
                value = self._clean_text(str(value))

            tmp_prompt += f"{label}: {value}\n"

        final_prompt = self.base_prompt + tmp_prompt
        return final_prompt.strip()

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

    def analyze(self, **kwargs) -> dict:
        sample_error_result = {
            "id": kwargs.get("Index", None),
            "classification": "Error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        try:
            prompt = self._create_prompt(**kwargs)
        except Exception as e:
            logger.warning(f"Error creating prompt for ad {kwargs.get('Index', 'UNKNOWN')}: {e}")
            return sample_error_result

        try:
            response = self._send_request(prompt)
        except Exception as e:
            logger.warning(f"Error sending request for ad {kwargs.get('Index', 'UNKNOWN')}: {e}")
            return sample_error_result

        try:
            result = self._parse_response(response)
        except Exception as e:
            logger.warning(f"Error parsing response for ad {kwargs.get('Index', 'UNKNOWN')}: {e}")
            return sample_error_result

        logger.debug(f"Ad {kwargs.get('Index', 'UNKNOWN')} classified as {result['classification']}")

        return {
            "id": kwargs.get("Index", None),
            "classification": result["classification"],
            "prompt_tokens": result["tokens"]["prompt_tokens"],
            "completion_tokens": result["tokens"]["completion_tokens"],
            "total_tokens": result["tokens"]["total_tokens"]
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
            "That is who Donald Trump and his allies are. Now, we must remember who we are and fight for the country we so love. Donate right away. #gotrump"
        ),
        "ad_creative_link_titles": ["Now or never"],
        "ad_delivery_start_time": "2024-10-28",
        "ad_delivery_stop_time": "2024-10-29",
        "bylines": "HARRIS VICTORY FUND",
        "transcript_translated": "you"
    }

    ad_analyzer = AdAnalyzer(model=model, temperature=temperature)
    result = ad_analyzer.analyze(**ad)
