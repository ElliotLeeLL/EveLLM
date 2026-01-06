from pathlib import Path

from tokenizers import Tokenizer

class GemmaTokenizer(Tokenizer):
    def __init__(self, tokenizer_file_path, model):
        tok_file = Path(tokenizer_file_path)
        self.tok_ = Tokenizer.from_file(str(tok_file))
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token

    def encode(self, text):
        return self.tok_.encode(text).ids

    def decode(self, token_ids):
        return self.tok_.decode(token_ids, skip_special_tokens=False)

def apply_chat_template(user_text):
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"