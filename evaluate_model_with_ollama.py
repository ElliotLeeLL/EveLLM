import json
import urllib.request
from datetime import datetime
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
            entry["score"] = score
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores, test_data

if __name__ == "__main__":
    result_data_path = Path("result_data") / "instruction_data_with_response_202508111754.json"
    with open(result_data_path, "r") as file:
        test_data = json.load(file)
    scores, result_data_with_score  = generate_model_scores(test_data, "model_response")
    scored_result_data_path = Path("result_data") / f"instruction_data_with_response_and_score_{datetime.now().strftime('%Y%m%d%H%M')}.json"
    with open(scored_result_data_path, "w") as file:
        json.dump(result_data_with_score, file, indent=4)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")