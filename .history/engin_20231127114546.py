import tiktoken
# Pakcage for converting tokenize a chars 
import torch    

# start 
enc = tiktoken.get_encoding("gpt2")             # encoding type 
print(enc.n_vocab)

# importing the datas
with open('input.txt', 'r', encoding='utf-8') as f:
    my_text = f.read()


print("length of dataset in characters: ", len(my_text))

# let's look at the first 1000 characters
print(my_text[:1000])

# encode a text
encoded_txt = enc.encode(my_text)

print("Encoded first 1000 char in encoded text is : ", encoded_txt[:1000])

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
sample_txt = "Very well; and could be content to give him good" # for train                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  N    M
print("0-> BEFORE ENCODE => new text is : ", sample_txt)

encode_str = encode(sample_txt)
print("1-> AFTER ENCODE => the new encoded is ", encode_str)

decoded_str = decode(encode_str)
print("2-> FINAL DECODE IT => the text is =", decoded_str)

# torch 
data = torch.tensor(encode(my_text), dtype=torch.long)

# show the data shape and data type
print("\nData Shape : ",data.shape, "\nData Type :", data.dtype)

# print tenser data left then 1000
print("tensored data left then 1000 only ",data[:1000])

# Lets now split up the data into train and validation sets
net1 = int(0.9 * len(data)) # first 90% will be train , rest => val
train_data = data[:net1]    # data to train is 90% of total datas

val_data = data[net1:]      # data to val is 10% of total datas

block_size = 8              # 8 digit chararacter(corresponfing to the vocab digit values) will be train to get 1 target next

train_data[:block_size+1]   # 8 inputs => char + 1 target]
print("tensor block size : ", train_data[:block_size+1])

# now lets train the data target when is gived context
# x is the input(context)
# y is the target 
# loop it in range of block_size and show the context and the target 
x = train_data[:block_size]
y = train_data[:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target : {target}")


#
# Manual torch using 
print("\n --- MANUAL torch feed to find next value using context  \n")
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # ix is the 
    x = torch.stack([data[i:i+block_size] for i in ix])       # 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # 
    
    return x, y

xb, yb = get_batch('train')  # 
print('inputs:')
print(xb.shape)  
print(xb)

print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target  = yb[b, t]
        print(f"when input is {context.tolist()} the target : {target}")


# -----
import Bigram
from bigran_lm import BigramLanguagBigramLanguageModel
print(" -- Using NN with Birgram Language Model -- ")

model = BigramLanguageModel(chars_vocab_size)

out = model(xb, yb)

print(out.shape)

