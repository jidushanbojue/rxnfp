import gzip
import pandas as pd
import numpy as np
from itertools import islice
from tqdm import tqdm, tqdm_notebook
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

df = pd.read_csv('../data/schneider50k.tsv', sep='\t')
df.head()

model, tokenizer = get_default_model_and_tokenizer('bert_ft_10k_25s')
ft_10k_rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
model, tokenizer = get_default_model_and_tokenizer('bert_ft')
ft_rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
model, tokenizer = get_default_model_and_tokenizer('bert_pretrained')
pretrained_rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)


fps_ft_10k = generate_fingerprints(df.rxn.values.tolist(), ft_10k_rxnfp_generator, batch_size=8)
np.savez_compressed('../data/fps_ft_10k', fps=fps_ft_10k)
print(fps_ft_10k.shape)

fps_ft_10k = np.load('../data/fps_ft_10k.npz')['fps']

fps_pretrained = generate_fingerprints(df.rxn.values.tolist(), pretrained_rxnfp_generator, batch_size=8)
np.savez_compressed('../data/fps_pretrained', fps=fps_pretrained)
print(fps_pretrained.shape)

fps_ft = generate_fingerprints(df.rxn.values.tolist(), ft_rxnfp_generator, batch_size=8)
np.savez_compressed('../data/fps_ft', fps=fps_ft)
print(fps_ft.shape)
