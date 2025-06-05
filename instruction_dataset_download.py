import json
import os
import urllib
from pathlib import Path

def down_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(text_data)
    else:
        with open(file_path, 'r', encoding="utf-8") as f:
            text_data = f.read()
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
    return data

if __name__ == '__main__':
    file_path = Path("instruction_data") / "instruction_data.json"
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    data = down_and_load_file(file_path, url)

    print("Entity number: ", len(data))