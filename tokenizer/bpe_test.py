import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
text_decoded = tokenizer.decode(ids)
print(ids)
print(text_decoded)

# Test code
unknown_word = "Akwirw ier"
unknown_ids = tokenizer.encode(unknown_word, allowed_special={"<|endoftext|>"})
print(unknown_ids)
for id in unknown_ids:
    print(id, "=>", tokenizer.decode([id]))
unknown_word_decoded = tokenizer.decode(unknown_ids)
print(unknown_word_decoded)