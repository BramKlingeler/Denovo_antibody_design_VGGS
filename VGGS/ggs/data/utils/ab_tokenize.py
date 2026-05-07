from typing import Union, Dict
from ggs.constants._tokens import TOKEN_GAP, TOKENS_AHO, ALPHABET_AHO
from ggs.constants._ranges_aho import AHO_FULL_RANGE

class GGS_AntibodyTokenizer:
    def __init__(self, chain_type: str = "H"):
        self.chain_type = chain_type
        self.encoder = ALPHABET_AHO

    def encode(self, sequence: str):
        return self.encoder.transform(sequence).tolist()

    def decode(self, indices):
        return "".join(self.encoder.inverse_transform(indices))


        
        
class Encoder(object):
    """convert between strings and their one-hot representations"""
    def __init__(self, alphabet = ALPHABET_AHO, chain_type: str = "H"):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)
    
    @property
    def vocab(self) -> np.ndarray:
        return np.array(list(self.alphabet))
    
    @property
    def tokenized_vocab(self) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in self.alphabet])

    def onehotize(self, batch):
        #create a tensor, and then onehotize using scatter_
        onehot = torch.zeros(len(batch), self.vocab_size)
        onehot.scatter_(1, batch.unsqueeze(1), 1)
        return onehot
    
    def encode(self, seq_or_batch: str or list, return_tensor = True) -> np.ndarray or torch.Tensor:
        if isinstance(seq_or_batch, str):
            encoded_list = [self.a_to_t[a] for a in seq_or_batch]
        else:
            encoded_list = [[self.a_to_t[a] for a in seq] for seq in seq_or_batch]
        return torch.tensor(encoded_list) if return_tensor else encoded_list
    
    def decode(self, x: np.ndarray or list or torch.Tensor) -> str or list:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, torch.Tensor):
            x = x.tolist()

        if isinstance(x[0], list):
            return [''.join([self.t_to_a[t] for t in xi]) for xi in x]
        else:
            return ''.join([self.t_to_a[t] for t in x])
            
            
            
def align_to_aho(sequence: Dict[int, str], chain_type: str = "H") -> str:
    if chain_type not in AHO_FULL_RANGE:
        raise ValueError(f"Unsupported chain_type: {chain_type}")

    full_seq = []
    for position in AHO_FULL_RANGE[chain_type]:
        aa = sequence.get(position, TOKEN_GAP)
        full_seq.append(aa)

    return "".join(full_seq)

# Tokenizer instance (module-level)
tokenizer = GGS_AntibodyTokenizer(chain_type="H")

def tokenize(seq_dict_or_str: Union[str, Dict[int, str]]):
    if isinstance(seq_dict_or_str, dict):
        aho_seq = align_to_aho(seq_dict_or_str, chain_type="H")
        return tokenizer.encode(aho_seq)
    else:
        return tokenizer.encode(seq_dict_or_str)