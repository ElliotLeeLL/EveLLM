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


if __name__ == "__main__":
    result_data_path = Path("result_data") / "instruction_data_with_response_eve-llm-qwen3-0P6B_202508222325-important.json"
    model_name = "eve-llm-qwen3-0P6B"
    with open(result_data_path, "r") as file:
        test_data = json.load(file)
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        response_text = entry["model_response"]
        response_text = response_text.split("</think>\n\n", 1)[1]
        response_text = response_text.removesuffix("<|im_end|>")
        entry["model_response"] = response_text
        test_data[i]["model_response"] = response_text
    result_data_path = Path(
        "result_data") / f"instruction_data_with_response_{model_name}_{datetime.now().strftime('%Y%m%d%H%M')}.json"
    with open(result_data_path, "w") as file:
        json.dump(test_data, file, indent=4)