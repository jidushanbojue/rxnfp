import torch
import pkg_resources
import numpy as np
from typing import List
from tqdm import tqdm
from itertools import islice


from transformers import BertModel

from rxnfp.core import (
    FingerprintGenerator
)
from rxnfp.tokenization import (
    SmilesTokenizer,
    convert_reaction_to_valid_features,
    convert_reaction_to_valid_features_batch,
)

class RXNBERTFingerprintGenerator(FingerprintGenerator):
    """
    Generate RXNBERT fingerprints from reaction SMILES
    """

    def __init__(self, model: BertModel, tokenizer: SmilesTokenizer, force_no_cuda=False):
        super(RXNBERTFingerprintGenerator).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")

    def convert(self, rxn_smiles: str):
        """
        Convert rxn_smiles to fingerprint

        Args:
            rxn_smiles (str): precursors>>products
        """
        bert_inputs = convert_reaction_to_valid_features(rxn_smiles, self.tokenizer)
        with torch.no_grad():
            output, _ = self.model(
                torch.tensor(bert_inputs.input_ids.astype(np.int64)).unsqueeze(0).to(self.device),
                torch.tensor(bert_inputs.input_mask.astype(np.int64)).unsqueeze(0).to(self.device),
                torch.tensor(bert_inputs.segment_ids.astype(np.int64)).unsqueeze(0).to(self.device),
            )

        # [CLS] token embeddings
        embeddings = output.squeeze()[0].cpu().numpy().tolist()
        return embeddings

    def convert_batch(self, rxn_smiles_list: List[str]):
        bert_inputs = convert_reaction_to_valid_features_batch(
            rxn_smiles_list, self.tokenizer
        )
        with torch.no_grad():
            output, _ = self.model(
                torch.tensor(bert_inputs.input_ids.astype(np.int64)).to(self.device),
                torch.tensor(bert_inputs.input_mask.astype(np.int64)).to(self.device),
                torch.tensor(bert_inputs.segment_ids.astype(np.int64)).to(self.device),
            )

        # [CLS] token embeddings in position 0
        embeddings = output[:, 0, :].cpu().numpy().tolist()
        return embeddings


class RXNBERTMinhashFingerprintGenerator(FingerprintGenerator):
    """
    Generate RXNBERT fingerprints from reaction SMILES
    """

    def __init__(
        self, model: BertModel, tokenizer: SmilesTokenizer, permutations=256, seed=42, force_no_cuda=False
    ):
        super(RXNBERTFingerprintGenerator).__init__()
        import tmap as tm

        self.model = model
        self.tokenizer = tokenizer
        self.minhash = tm.Minhash(model.config.hidden_size, seed, permutations)
        self.generator = RXNBERTFingerprintGenerator(model, tokenizer)
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")

    def convert(self, rxn_smiles: str):
        """
        Convert rxn_smiles to fingerprint

        Args:
            rxn_smiles (str): precursors>>products
        """
        float_fingerprint = self.generator.convert(rxn_smiles)
        minhash_fingerprint = self.minhash.from_weight_array(
            float_fingerprint, method="I2CWS"
        )
        return minhash_fingerprint

    def convert_batch(self, rxn_smiles_list: List[str]):
        float_fingerprints = self.generator.convert_batch(rxn_smiles_list)
        minhash_fingerprints = [
            self.minhash.from_weight_array(fp, method="I2CWS")
            for fp in float_fingerprints
        ]
        return minhash_fingerprints

def get_default_model_and_tokenizer(model='bert_ft', force_no_cuda=False):

    model_path =  pkg_resources.resource_filename(
                "rxnfp",
                f"models/transformers/{model}"
            )

    tokenizer_vocab_path = (
        pkg_resources.resource_filename(
                    "rxnfp",
                    f"models/transformers/{model}/vocab.txt"
                )
    )
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")

    model = BertModel.from_pretrained(model_path)
    model = model.eval()
    model.to(device)

    tokenizer = SmilesTokenizer(
        tokenizer_vocab_path, max_len=model.config.max_position_embeddings
    )
    return model, tokenizer

def generate_fingerprints(rxns: List[str], fingerprint_generator:FingerprintGenerator, batch_size=1) -> np.array:
    fps = []

    n_batches = len(rxns) // batch_size
    emb_iter = iter(rxns)
    for i in tqdm(range(n_batches)):
        batch = list(islice(emb_iter, batch_size))

        fps_batch = fingerprint_generator.convert_batch(batch)

        fps += fps_batch
    return np.array(fps)

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

example_rxn = "Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"

fp = rxnfp_generator.convert(example_rxn)
print(len(fp))
print(fp[:5])

fps = rxnfp_generator.convert_batch([example_rxn, example_rxn])
print(len(fps), len(fps[0]))

model, tokenizer = get_default_model_and_tokenizer()

rxnmhfp_generator = RXNBERTMinhashFingerprintGenerator(model, tokenizer)

example_rxn = "Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"

fp = rxnmhfp_generator.convert(example_rxn)
print(len(fp))
print(fp[:5])
