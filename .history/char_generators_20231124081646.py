import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)

# encode a text
my_text = "Salam Aleykoum monsieur ali"
encoded_txt = enc.encode(my_text)

print(encoded_txt)

#decode

