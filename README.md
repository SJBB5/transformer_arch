DOWNLOADING DATA:


HOW TO RUN:
run uv sync
download dataset from this google drive link and put it in the data folder https://docs.google.com/document/d/1qF6ajBskjxUzme8L7TieEQkimGch5cO-ky5yFVj3R4Q/edit?usp=sharing
train: python train.py
generate: python generate.py

RESULTS:
The loss curve is automatically saved to this directory after training (view it as loss_curve.png).

Sample output from trained model:
hey franklin what is up dude n t now some sticky s been lessons s now ? okay anywhere ? trevor to property a little wet . it break , the without . i spend , south . enforcement depressing . oh , about your easy . hey keep the pilot . bonded girl our i ve been them to oh on pretty lost , tatavia , i d secret the inbred a pointing at nervous where destroying missed could so night a action only you our bad . i m a sitting


WRITEUP:
Our dataset was 121 pages of voicelines from Trevor in GTA V. The model takes about 10 minutes to train and is not very coherent. At first we used a character tokenizer and this did not work very well. There was a noticeable improvement when switching over to a word tokenizer. If we had more time I would try and get a more coherent dataset. A collection of random voicelines especially from someone like Trevor doesn't make a good model. 
CONTRIBUTIONS:
We worked equally on the implementation. In class, we worked on one computer and contributed equally to the implementation. The work was moved to this git repo recently just to organize our work better; the commits on this repo is not indicative of the contributions to the project.

