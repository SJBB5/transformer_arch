DOWNLOADING DATA:
download dataset from this Google Drive link and put it in the data folder https://docs.google.com/document/d/1qF6ajBskjxUzme8L7TieEQkimGch5cO-ky5yFVj3R4Q/edit?usp=sharing as "AllTrevorVoicelines.txt"


HOW TO RUN:

uv sync

train: uv run python train.py

generate: uv run python generate.py

RESULTS:
The loss curve is automatically saved to this directory after training (view it as loss_curve.png).

WRITEUP:
Our dataset was 121 pages of voicelines from Trevor in GTA V. The model takes about 10 minutes to train and is not very coherent. At first we used a character tokenizer and this did not work very well. There was a noticeable improvement when switching over to a word tokenizer. If we had more time I would try and get a more coherent dataset. A collection of random voicelines especially from someone like Trevor doesn't make a good model. 

We used a context length of 32, which we deemed optimal for the short nature of trevor's phrases. We used 6 transformer blocks, d_model of 512, and d_hidden of 1024. Given more time we would like to implement batching, as that would likley improve our model significantly.


CONTRIBUTIONS:
We worked equally on the implementation. In class, we worked on one computer and contributed equally to the implementation. The work was moved to this git repo recently just to organize our work better; the commits on this repo is not indicative of the contributions to the project.

