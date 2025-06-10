import json
import urllib.request
from pathlib import Path
from tqdm import tqdm

from utils.model_utils import format_input_alpaca


def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11434/api/chat",
):
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "options": {
            "seed": 43,
            "temperature": 0.0,
            "num_ctx": 2048
        }
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url=url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data = response_data + response_json["message"]["content"]
    return response_data

def generate_model_scores(
        json_data,
        json_key,
        model="llama3"
):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input_alpaca(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}` "
            f"on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(float(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores

if __name__ == "__main__":
    result_data_path = Path("result_data") / "instruction_data_with_response.json"
    with open(result_data_path, "r") as file:
        test_data = json.load(file)
    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")