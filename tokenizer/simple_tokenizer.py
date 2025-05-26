import urllib.request
import re


# # Test code
# url = (
#     "https://raw.githubusercontent.com/rasbt/"
#     "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#     "the-verdict.txt"
# )
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)

# with open(file_path, "r", encoding="utf-8") as f:
#     raw_text = f.read()
# print("Total number of characters:", len(raw_text))
# print(raw_text[:99])

# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item for item in preprocessed if item.strip()]

# print("Total number of items:", len(preprocessed))
# print(preprocessed[:99])

# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)
# print("Vocabulary size:", vocab_size)

# vocab = {token: integer for integer, token in enumerate(all_words)}
# for token, integer in vocab.items():
#     if(integer >30):
#         break
#     print(token, "=>", integer)

class SimpleTokenizer:
    def __init__(self):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        file_path = "../the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item for item in preprocessed if item.strip()]
        all_words = sorted(set(preprocessed))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token: integer for integer, token in enumerate(all_words)}
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encoder(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decoder(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

tokenizer = SimpleTokenizer()
ids = tokenizer.encoder(text)
text = tokenizer.decoder(ids)
print(text)