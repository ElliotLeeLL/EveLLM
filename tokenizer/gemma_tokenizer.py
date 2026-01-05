from pathlib import Path

from tokenizers import Tokenizer

class GemmaTokenizer(Tokenizer):
    def __init__(self, tokenizer_file_path, model):
        tok_file = Path(tokenizer_file_path)
        self.tok_ = Tokenizer.from_file(str(tok_file))
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token