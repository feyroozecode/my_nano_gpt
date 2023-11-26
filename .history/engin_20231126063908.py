import tiktoken
# Pakcage for converting tokenize a chars 
import torch    

# start 
enc = tiktoken.get_encoding("gpt2")             # encoding type 
print(enc.n_vocab)

# encode a text
encoded_txt = enc.encode(my_text)

print("Encoded text is : ", encoded_txt)

#decode
decoded_txt = enc.decode(encoded_txt)
print("Decoded text is : ", decoded_txt)

chars_lngtext = sorted(list(set(my_text)))  # sorted list by chars 
chars_vocab_size = len(chars_lngtext)       # vocab words char length
print("chars of my text : ", chars_lngtext , "\nand size is ", chars_vocab_size, "chars ")

joined_text = ''.join(chars_lngtext)        # Joined chars 
print("joinded text is : ", joined_text)  

# A dictionary where characters are keys, and their corresponding indices (integers) are values.
stoi = {
    char:i for i,char in enumerate(chars_lngtext)
}

# 'itos' is the inverse dictionary where indices are keys, and the characters are values.
itos = {
    i: char for i, char in enumerate(chars_lngtext)
}

# 'encode' is a lambda function that takes a string 's' and returns a list of integers.
# It uses list comprehension to map each character in the string to its corresponding integer using the 'stoi' dictionary.
encode = lambda s: [stoi[c] for c in s ]

# 'decode' is a lambda function that takes a list of integers 'l' and returns a string.
# It uses list comprehension to map each integer in the list to its corresponding character using the 'itos' dictionary,
# and then joins the characters into a string using the ''.join() method
decode = lambda list: ''.join([itos[i] for i in list ])

# test encode and decode
sample_txt = "bonjour monsieur ali" # for train                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  N    M
print("0-> BEFORE ENCODE => new text is : ", sample_txt)

encode_str = encode(sample_txt)
print("1-> AFTER ENCODE => the new encoded is ", encode_str)

decoded_str = decode(encode_str)
print("2-> FINAL DECODE IT => the text is =", decoded_str)

# torch 
data = torch.tensor(encode(my_text), dtype=torch.long)

print(data.shape, data.dtype)

print(data[:1000])