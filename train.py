import torch
import random
import matplotlib.pyplot as plt
from src.config import Config
from src.tokenizer import Tokenizer, process_text
from src.model import Transformer

def get_batch(data, n_ctx):
    # pick random starting point
    i = random.randint(0, len(data) - n_ctx - 1)

    # extract sequences
    x = data[i : i + n_ctx]
    y = data[i + 1 : i + n_ctx + 1]

    return x, y

# Get the data
with open("data/AllTrevorVoicelines.txt") as f:
    book = f.read()

# use profs processing function
processed = process_text(book)

# tockenize the data
tokenizer = Tokenizer(processed)

# config
cfg = Config(d_vocab=tokenizer.vocab_size, n_ctx=32)

# convert to tensor, so pass into function works
full_data_tensor = torch.tensor(tokenizer.encode(processed), dtype=torch.long)

model = Transformer(cfg)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
loss_list = []
iter_list = []

#  Loop
for i in range(5000):
    # get sequence
    xb, yb = get_batch(full_data_tensor, n_ctx=cfg.n_ctx)
    
    # forward pass
    logits = model(xb) 
    
    # Loss calculatiion
    loss = criterion(logits, yb)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"{i}: Loss {loss.item()}")
        loss_list.append(loss.item())
        iter_list.append(i)

# Plotting loss
plt.plot(iter_list, loss_list)
plt.ylim(bottom=0)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("loss_curve.png") # Saves for GitHub
plt.show()

torch.save(model.state_dict(), "model.pt")

