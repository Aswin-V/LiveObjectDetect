import base64
import json
import re
import logging
import requests
import cv2
import numpy as np
from .analyzer_base import Analyzer

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

class GeminiAnalyzer(Analyzer):
    """Analyzer using the Gemini model."""

    def __init__(self, api_key: str, prompt: str):
        if not api_key:
            raise ValueError("Gemini API Key is missing.")
        self.api_key = api_key
        self.prompt = prompt

    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Sends a single frame to the Gemini API for analysis.
        """
        logging.info("Attempting Gemini analysis.")
        
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            logging.error("Failed to encode frame to JPEG.")
            return None
        
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "contents": [{"parts": [{"text": self.prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        
        headers = {"Content-Type": "application/json"}
        full_api_url = f"{GEMINI_API_URL}?key={self.api_key}"

        try:
            logging.info("Sending request to Gemini API.")
            response = requests.post(full_api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            logging.info("Received successful response from Gemini API.")
            result = response.json()
            
            if (result.get("candidates") and result["candidates"][0].get("content") and 
                result["candidates"][0]["content"].get("parts")):
                json_text = result["candidates"][0]["content"]["parts"][0]["text"]
                # Use a regex to find the JSON block, robust against ```json, ```, etc.
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_text)
                if match:
                    json_text = match.group(1)
                logging.info("Successfully parsed Gemini response.")
                return json.loads(json_text)
            else:
                logging.warning(f"No valid content in Gemini API response: {result}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Gemini API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from Gemini response: {e}. Response text: {response.text}")
            return None