import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.n_vocab)

# encode a text
my_text = "Salam Aleykoum bonjour monsieur ali"
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

# 'itos' is the inverse dictionary where indices are keys, and the characters are values.
itos = {
    i: char for i, ch in enumerate(chars_lngtext)
}

# 'encode' is a lambda function that takes a string 's' and returns a list of integers.
# It uses list comprehension to map each character in the string to its corresponding integer using the 'stoi' dictionary.
encode = lambda s: [stoi[c] for c in s ]

# 'decode' is a lambda function that takes a list of integers 'l' and returns a string.
# It uses list comprehension to map each integer in the list to its corresponding character using the 'itos' dictionary,
# and then joins the characters into a string using the ''.join() method
decode = lambda list ''.join([itos] for i in list )

# test encode and decode 

encode_str = endcode("bonjour monsieur ali")
print(encode_str)