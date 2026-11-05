"""Save the models available to the configured Groq account as output.json."""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set. Add it to the project .env file.")

    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    output_path = Path(__file__).resolve().parents[1] / "output.json"
    output_path.write_text(
        json.dumps(response.json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Model list saved to {output_path}")


if __name__ == "__main__":
    main()
