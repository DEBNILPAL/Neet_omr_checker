import argparse
import json
import os
import sys
from PIL import Image
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

from omr_processor import process_omr_image
from ai_analyzer import analyze_with_gemini

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", -1: None, -2: None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--use-ai", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    pil_img = Image.open(args.image)
    detected = None

    if args.use_ai:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            result = analyze_with_gemini(pil_img, gemini_key)
            if "error" not in result:
                detected = result

    if detected is None:
        detected = process_omr_image(pil_img, allow_multi=False)

    answers = []
    for col_num in range(1, 4):
        col_answers = detected.get(f"col_{col_num}", [-1] * 50)
        for q_idx, ans_idx in enumerate(col_answers):
            answers.append(
                {
                    "questionNumber": (col_num - 1) * 50 + q_idx + 1,
                    "selectedOption": INDEX_TO_LETTER.get(ans_idx),
                }
            )

    print(json.dumps(answers))


if __name__ == "__main__":
    main()
