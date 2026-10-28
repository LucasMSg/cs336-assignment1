import os
import regex as re
from typing import BinaryIO, Dict, List, Tuple
from multiprocessing import Pool
from collections import Counter, defaultdict

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    print("file size", file_size)
    file.seek(0)
    
    chunk_size = file_size // desired_num_chunks
    print("chunk size", chunk_size)
    
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    print("chunk_boundaries", chunk_boundaries)
    
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(
    input_path: str | os.PathLike, 
    start: int, 
    end: int, 
    special_tokens: List[str]
) -> Counter:

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_text = (f.read(end - start).decode("utf-8", errors="ignore"))
        
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    pieces = re.split(pattern, chunk_text)
        
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    counts = Counter()
    for piece in pieces:
        for m in re.finditer(PAT, piece):
            token_bytes = m.group().encode("utf-8")
            # Represent each pre-token as a tuple of single bytes,
            # e.g. b" cat" -> (b" ", b"c", b"a", b"t")
            counts[tuple(bytes([b]) for b in token_bytes)] += 1
    return counts
        
        
    # Inside, it opens the file, 
    # seeks to start, reads end - start bytes, decodes, 
    # splits on special tokens, runs finditer, and returns a Counter of pre-tokens. 
    # It must be defined at module level, not inside train_bpe, so it can be pickled.

def count_pairs(word_counts):
    pairs = Counter()
    for word, freq in word_counts.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs

# NEED TO AVOID EXECUTING APPLY MERGE FOR ALL WORDS
def apply_merge(word, pair, new_token):
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_token)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    #vocabulary initialization
    vocabulary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    print("vocabulary",vocabulary)
    
    #adding special tokens 
    for token in special_tokens:
        vocabulary[len(vocabulary)] = token.encode("utf-8")
        
    #chunking with find_chunk_boundaries from pretokenization_example.py
    start_and_end = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
        print("boundaries", boundaries)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start_and_end.append([start, end])
        '''for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))'''
            
    print(start_and_end)
    #pretokenization
    num_processes = os.cpu_count()
    print("num of processes", num_processes)
    
    
    items = []
    for pair in start_and_end:
        items.append((input_path, pair[0], pair[1], special_tokens))
    
    counterTotal = Counter()
    with Pool(processes=num_processes) as p:
        for c in p.starmap(pretokenize, items):
            counterTotal.update(c)
    
    words = [w for w in counterTotal]           # tuples of bytes
    freqs = [counterTotal[w] for w in words] 
    
    # Count pairs ONCE, and index which words contain each pair
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    
    
    for idx, (word, freq) in enumerate(zip(words, freqs)):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(idx)
    
    
    merges = []
    
    while len(vocabulary) < vocab_size and pair_counts:
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        new_token = best[0] + best[1]
        merges.append(best)
        vocabulary[len(vocabulary)] = new_token

        # Only touch words that actually contain tokens merged
        for idx in list(pair_to_words.pop(best, ())):
            word, freq = words[idx], freqs[idx]

            # 1. remove this word's old pairs
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair in pair_to_words:
                    pair_to_words[pair].discard(idx)

            # 2. merge inside this word only
            new_word = apply_merge(word, best, new_token)
            words[idx] = new_word

            # 3. add its new pairs
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(idx)

    return vocabulary, merges
