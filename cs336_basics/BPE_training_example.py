from collections import Counter
# get text 
with open("../data/BPE_example.txt", encoding='utf-8') as f:
    text = f.read()


# We initialize our vocabulary with our special token <|endoftext|> and the 256 byte values
vocabulary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
vocabulary[len(vocabulary)] = b"<|endoftext|>"

# Pre-tokenization
''' For simplicity and to focus on the merge procedure, we assume in this example that pre-
    tokenization simply splits on whitespace.'''
    
# just splitting text by word, pretokenize here is simply splits on whitespace
splitted = text.split() 

# map of words to their incidence
preTokenized = {}
for token in splitted:
    key = token.encode("utf-8")
    preTokenized[key] = preTokenized.get(key, 0) + 1
# print(preTokenized)
# {b'low': 5, b'lower': 2, b'widest': 3, b'newest': 6}

#Merges
''' We first look at every successive pair of bytes and sum the frequency of the words where they
    appear { lo : 7, ow : 7, we : 8, er : 2, wi : 3, id : 3, de : 3, es : 9, st : 9, ne : 6, ew : 6}. ''' 

def count_pairs(word_freqs):
    pairs = Counter()
    for symbols, count in word_freqs.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += count
    return pairs

word_freqs = {
    tuple(bytes([b]) for b in word): count
    for word, count in preTokenized.items()
}

def merge_word(symbols, pair):
    new_symbols = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            new_symbols.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return tuple(new_symbols)

pairForMerge = count_pairs(word_freqs)
   
merges = []
for _ in range(6):
    pairForMerge = count_pairs(word_freqs)
    if not pairForMerge:
        break

    best_pair = max(pairForMerge, key=lambda pair: (pairForMerge[pair], pair))
    merges.append(best_pair)
    vocabulary[len(vocabulary)] = best_pair[0] + best_pair[1]

    word_freqs = {merge_word(w, best_pair): c for w, c in word_freqs.items()}
    print(best_pair, word_freqs)
    
    
 #print(vocabulary)
    