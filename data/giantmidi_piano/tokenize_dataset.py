from miditok import REMI
from pathlib import Path
import json
import numpy as np
import math
import pickle
import os

# config
vocab_size = 1000
train_split = 0.9

# midi_root = "sample_midis"
midi_root = "midis"

# Creates the tokenizer and list the file paths
tokenizer = REMI()  # using defaults parameters (constants.py)
midi_paths = list(Path(midi_root).glob("**/*.mid"))

# A validation method to discard MIDIs we do not want
# It can also be used for custom pre-processing, for instance if you want to merge
# some tracks before tokenizing a MIDI file
def midi_valid(midi) -> bool:
    # if any(ts.numerator != 4 for ts in midi.time_signature_changes):
        # return False  # time signature different from 4/*, 4 beats per bar
    return True

# Builds the vocabulary with BPE
tokenizer.learn_bpe(vocab_size=vocab_size, files_paths=midi_paths)


n_midi_files = len(midi_paths)
n_train_files = int(math.floor(train_split*n_midi_files))
train_midis = midi_paths[:n_train_files]
valid_midis = midi_paths[n_train_files:]

tokenizer.tokenize_midi_dataset(        # 2 velocity and 1 duration values
    train_midis,
    Path(midi_root, "train_tokens"),
    midi_valid,
)

tokenizer.tokenize_midi_dataset(        # 2 velocity and 1 duration values
    valid_midis,
    Path(midi_root, "valid_tokens"),
    midi_valid,
)

# now write the jPath("sample_midis/tokens")son tokens to bin files for faster loading
train_tokens = []
for json_file in Path(midi_root, "train_tokens").glob("**/*.json"):
    data = json.load(open(json_file, "r"))["ids"][0]
    print(json_file, data[:10])
    train_tokens.extend([tokenizer["BOS_None"], *data, tokenizer["EOS_None"]])

np.array(train_tokens, dtype=np.int16).tofile(Path(midi_root, "train.bin"))

# now write the jPath("sample_midis/tokens")son tokens to bin files for faster loading
valid_tokens = []
for json_file in Path(midi_root, "valid_tokens").glob("**/*.json"):
    data = json.load(open(json_file, "r"))["ids"][0]
    print(json_file, data[:10])
    valid_tokens.extend([tokenizer["BOS_None"], *data, tokenizer["EOS_None"]])

np.array(valid_tokens, dtype=np.int16).tofile(Path(midi_root, "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    # 'itos': itos,
    # 'stoi': stoi,
}
with open(os.path.join(midi_root, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
