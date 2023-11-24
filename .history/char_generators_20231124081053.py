import tiktoken

enc = tiktoken.get_encoding("gpt2")
enc.n_vocab
enc.decode([1, 2, 3, 4])