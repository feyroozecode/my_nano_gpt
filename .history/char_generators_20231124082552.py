import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)

# encode a text
my_text = "Salam Aleykoum monsieur ali"
encoded_txt = enc.encode(my_text)

print("Encoded text is : ", encoded_txt)

#decode
decoded_txt = enc.decode(encoded_txt)
print("Decoded text is : ", decoded_txt)

chars_lngtext = sorted(list(set(my_text)))
chars_vocab_size = len(chars_lngtext)
print("chars of my text : ", chars_lngtext , "\nand size is ", chars_vocab_size, "chars ")

joined_text = ''.join(chars_lngtext)
print("joinded text is : ", joined_text)

# A dictionary where characters are keys, and their corresponding indices (integers) are values.
stoi = {
    char: i for i, ch in enumerate(chars_lngtext)
}

