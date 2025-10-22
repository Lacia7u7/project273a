from data import mappings, vocab
from graph import builder

import pandas as pd

def test_unknown_inclusion():
    df = pd.DataFrame({'col': ['A','B','A','C', None]})
    vocab_dict = vocab.build_vocab(df, 'col', add_unknown=True, unknown_label="UNKNOWN")
    assert "UNKNOWN" in vocab_dict
    # any new value should map to UNKNOWN
    new_val = "Z"
    idx = vocab_dict.get(new_val, vocab_dict.get("UNKNOWN"))
    assert idx == vocab_dict.get("UNKNOWN")
