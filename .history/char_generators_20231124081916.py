import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)

# encode a text
my_text = "Salam Aleykoum monsieur ali"
encoded_txt = enc.encode(my_text)

print("Encoded text is ", encoded_txt)

#decode
decoded_txt = enc.decode(encoded_txt)
print(decoded_txt)

