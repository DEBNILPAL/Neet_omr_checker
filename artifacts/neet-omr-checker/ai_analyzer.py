"""
AI-Powered OMR Analysis Module
Uses Gemini or OpenRouter APIs as fallback for better OMR detection.
"""

import base64
import json
import re
from io import BytesIO
from PIL import Image
import requests


def encode_image_to_base64(image_input):
    """Encode PIL Image or bytes to base64 string."""
    if isinstance(image_input, Image.Image):
        buffer = BytesIO()
        image_input.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    elif hasattr(image_input, 'read'):
        image_input.seek(0)
        data = image_input.read()
        image_input.seek(0)
        return base64.b64encode(data).decode("utf-8")
    return None


OMR_ANALYSIS_PROMPT = """
You are an expert OMR (Optical Mark Recognition) analyzer for NEET exam sheets.

Analyze this OMR sheet image carefully. The sheet has 4 columns of answers, each with 50 questions (rows).
Each question has 4 options: A, B, C, D (bubbles arranged left to right).

For each column (1-4) and each question (1-50), identify:
- Which bubble is filled/darkened (A, B, C, or D)
- If no bubble is filled, mark as "unattempted"
- If multiple bubbles are filled, mark as "multiple"

Return ONLY a valid JSON response in this exact format:
{
  "col_1": ["A", "B", "C", "D", "-", "A", ...],  (50 entries, - means unattempted)
  "col_2": ["B", "A", "-", "C", ...],
  "col_3": ["D", "C", "B", "A", ...],
  "col_4": ["A", "-", "D", "B", ...]
}

Use exactly these values:
- "A", "B", "C", "D" for filled bubbles
- "-" for unattempted
- "M" for multiple bubbles filled

Be very precise and analyze the actual filled (darkened) bubbles carefully.
Each column should have exactly 50 entries.
"""


def analyze_with_gemini(image_input, api_key, col_range=None):
    """
    Use Google Gemini API to analyze OMR sheet.
    
    Args:
        image_input: PIL Image or file-like object
        api_key: Google Gemini API key
        col_range: optional tuple (start_col, end_col) to analyze specific columns
    
    Returns:
        dict with col_1..col_4 answers or error message
    """
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        if isinstance(image_input, Image.Image):
            pil_image = image_input
        else:
            image_input.seek(0)
            pil_image = Image.open(image_input)
            image_input.seek(0)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content([OMR_ANALYSIS_PROMPT, pil_image])
        
        text = response.text.strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return _parse_ai_response(data)
        
        return {'error': 'Could not parse Gemini response', 'raw': text}
    
    except Exception as e:
        return {'error': f'Gemini API error: {str(e)}'}


def analyze_with_openrouter(image_input, api_key, model="google/gemini-flash-1.5"):
    """
    Use OpenRouter API to analyze OMR sheet.
    
    Args:
        image_input: PIL Image or file-like object
        api_key: OpenRouter API key
        model: model to use (default: gemini-flash)
    
    Returns:
        dict with col_1..col_4 answers or error message
    """
    try:
        img_b64 = encode_image_to_base64(image_input)
        if not img_b64:
            return {'error': 'Could not encode image'}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://neet-omr-checker.app",
            "X-Title": "NEET OMR Checker"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": OMR_ANALYSIS_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            return {'error': f'OpenRouter API error: {response.status_code} - {response.text}'}
        
        result = response.json()
        text = result['choices'][0]['message']['content'].strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return _parse_ai_response(data)
        
        return {'error': 'Could not parse OpenRouter response', 'raw': text}
    
    except Exception as e:
        return {'error': f'OpenRouter API error: {str(e)}'}


def _parse_ai_response(data):
    """Convert AI response letters to 0-3 indices."""
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '-': -1, 'M': -2}
    
    result = {}
    for col_key in ['col_1', 'col_2', 'col_3', 'col_4']:
        if col_key in data:
            answers = []
            for ans in data[col_key][:50]:
                idx = letter_to_idx.get(str(ans).upper(), -1)
                answers.append(idx)
            while len(answers) < 50:
                answers.append(-1)
            result[col_key] = answers
        else:
            result[col_key] = [-1] * 50
    
    return result


def get_available_openrouter_models():
    """Return list of vision-capable models on OpenRouter."""
    return [
        "google/gemini-flash-1.5",
        "google/gemini-pro-1.5",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-3.2-90b-vision-instruct",
    ]
