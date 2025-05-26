import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class AdamDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = AdamDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# Test code
# with open("../the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
#
# max_length = 4
# vocab_size = 50257
# output_dim = 256
#
# dataloader = create_dataloader(
#     raw_text,
#     batch_size=8,
#     max_length=max_length,
#     stride=max_length,
#     shuffle=True,
#     drop_last=True,
#     num_workers=0
# )
#
# data_iter = iter(dataloader)
#
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# inputs, targets = next(data_iter)
# print("The shape of inputs: ")
# print(inputs.shape)
# token_embedded = token_embedding_layer(inputs)
# print("The shape of embedded inputs: ")
# print(token_embedded.shape)
# pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
# pos_token_embedded = pos_embedding_layer(torch.arange(max_length))
# print("The shape of embedded positional inputs: ")
# print(pos_token_embedded.shape)