import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from utils.Tokenizer import Tokenizer
from GPT import GPT
import numpy as np
import random
from tensorflow import keras

vocab_size = 20006  # Only consider the top 20k words
maxlen = 400  # Max sequence size
embed_dim = 512  # Embedding size for each token
num_heads = 3  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

tokenizer = Tokenizer(lower=False, reserved_symbols=["<EOA>"]) # initialize tokenizer with a reserved symbol <EOA> (End of Answer), and 

if not os.path.exists("./model/vocab.json"):
    print("No vocab cache found, generating cocab list...")
tokenizer.load_vocab("./model/vocab.json")

model = GPT(vocab_size, maxlen=maxlen, embed_dim=embed_dim, num_heads=num_heads, feed_forward_dim=feed_forward_dim)

model.load_weights('./model/model_chat_epoch_20.h5')#/saved_models/model_chat_epoch_10.h5')




def sample_from(logits):
    logits, indices = tf.math.top_k(logits, k=4, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)

text = ""
while True:
    max_tokens = 40
    start_prompt = text + "User : " + input("\nUser: ") + "\nmedGPT : "
    text += start_prompt + "\n"
    print("MedGPT:", end=" ")
    start_tokens = tokenizer.toSequence(start_prompt)
    num_tokens_generated = 40
    start_tokens = [_ for _ in start_tokens]
    num_tokens_generated = 0
    tokens_generated = []


    while num_tokens_generated <= max_tokens:
        pad_len = maxlen - len(start_tokens)
        sample_index = len(start_tokens) - 1
        if pad_len < 0:
            x = start_tokens[:maxlen]
            sample_index = maxlen - 1
        elif pad_len > 0:
            x = start_tokens + [0] * pad_len
        else:
            x = start_tokens
        x = np.array([x])
        y, _ = model.predict(x, verbose=0)
        sample_token = sample_from(y[0][sample_index])
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        if sample_token != 0:
            print(tokenizer.index2word[str(sample_token)], end=" ")
        else:
            break
        num_tokens_generated = len(tokens_generated)