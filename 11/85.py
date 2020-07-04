import torch
from torch import nn

word_to_ix = {
    "hello": 0,
    "world": 1
}
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
embeds = nn.Embedding(2, 5)
# word feature representation
hello_embed = embeds(lookup_tensor)
print(hello_embed)
