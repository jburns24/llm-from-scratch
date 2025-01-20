# import re
import os
import tiktoken
import torch
from torch.utils.data.dataloader import DataLoader

from gpt.gptdatasetv1 import GPTDatasetV1

# import hello_world.tokenizerv1
# import hello_world.tokenizerv2

# Creates our sliding window dataset off some given input text.
# Used to encode context from a document. Setting max_length equal to
# stride will result in batches fully utilizing the dataset (no skipped words)
# while including no overlap. More overlap means more overfitting on the model (bais)
#
# txt => holds the text to encode context from
## HYPER_PARMETER
# batch_size => dimension of each batch. Lower numbers == less memory to train but noisier model updates
## END_HYPER_PARAMETER
# max_length => length of the sliding window
# stride => how much window moves each iteration
# shuffle => if True have data reshuffled
# drop_last => if True last batch is dropped if it is shorter than batch_size.
# This ensures that batech always align perfectly. preventing loss spikes while training
# num_workers => number of CPU processes to use for preprocessing
#
# Example of encoded context
# [290] ----> 4920
# [290, 4920] ----> 2241
# [290, 4920, 2241] ----> 287
# [290, 4920, 2241, 287] ----> 257
def create_dataloader_v1(
    txt: str,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataset = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataset


data_dir = os.environ.get("DATA_DIR")

if data_dir == None:
    print("DATA_DIR not set defaulting to ../data")
    data_dir = "./data"

with open(f"{data_dir}/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Build vocabulary. Only needed if you are using your own tokenizer.
# TokenizerV1 and TokenizerV2 exmaples need this.
# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]

# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# vocab = {token: integer for integer, token in enumerate(all_tokens)}


# TokenizerV1
# text = """"It's the last he painted, you know,"
#         Mrs. Gisburn said with pardonable pride."""

# t = tokenizerv1.SimpleTokenizerV1(vocab)

# TokenizerV2
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))

# t = tokenizerv2.SimpleTokenizerV2(vocab)

# GPT-2 Tokenizer
# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#      " of someunknownPlace."
# )
# t = tiktoken.get_encoding("gpt2")
# ids = t.encode(text, allowed_special={'<|endoftext|>'})

# GPT-2 Tokeinzer on 'The Verdict'
t = tiktoken.get_encoding("gpt2")
ids = t.encode(raw_text)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

vocab_size = 50257
output_dim = 256


# torch.manual_seed(123)
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# token_embeddings = token_embedding_layer(inputs)

# context_length = max_length
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings[0])

# input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings[0])
