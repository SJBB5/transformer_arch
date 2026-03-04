import torch
from src.config import Config
from src.tokenizer import Tokenizer, process_text
from src.model import Transformer

# Load processed data to get tokenizer
with open("data/AllTrevorVoicelines.txt") as f:
    text = f.read()
processed = process_text(text)
tokenizer = Tokenizer(processed)

cfg = Config(d_vocab=tokenizer.vocab_size, n_ctx=32)
model = Transformer(cfg)

model.load_state_dict(torch.load("model.pt"))
model.eval()

prompt = "hey franklin what is up dude"
output = model.generate(prompt, 100, tokenizer, cfg)
print(output)