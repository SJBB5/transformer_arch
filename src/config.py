from dataclasses import dataclass

@dataclass
class Config:
    d_vocab: int  # get using vocab size
    n_ctx: int  # the maximum length of text the model can "see" at once
    d_model: int = 512  # size of embedding vectors
    d_hidden: int = 1024 # hidden layers
    n_layers: int = 4 # Number of Transformer blocks