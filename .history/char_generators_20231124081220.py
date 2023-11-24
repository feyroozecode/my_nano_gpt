import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)

# encode a text
my_text = "Sala Aleykoum bonjour monsieur ali"

enc.encode("Hello world!")