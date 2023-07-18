import asyncio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from utils.Tokenizer import Tokenizer
from GPT import GPT
import numpy as np
import random
from tensorflow import keras
import websockets
 

active_connections = set()

vocab_size = 20006  # Only consider the top 20k words
maxlen = 400  # Max sequence size
embed_dim = 512  # Embedding size for each token
num_heads = 3  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

tokenizer = Tokenizer(lower=False, reserved_symbols=["<EOA>"]) # initialize tokenizer with a reserved symbol <EOA> (End of Answer), and 

if not os.path.exists("./model/vocab.json"):
    print("No vocab cache found, generating cocab list...")
    while True:
        input()
tokenizer.load_vocab("./model/vocab.json")

model = GPT(vocab_size, maxlen=maxlen, embed_dim=embed_dim, num_heads=num_heads, feed_forward_dim=feed_forward_dim)

model.load_weights('./model/final_model.h5')#/saved_models/model_chat_epoch_10.h5')

 
def decode(logits):
    logits, indices = tf.math.top_k(logits, k=4, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)

async def calc(x, websocket):
    print(x)
    text = x
    max_tokens = 40
    start_prompt = text
    start_tokens = tokenizer.toSequence(start_prompt)
    if len(start_tokens) > 499:
        websocket.send("<OVER>")
        websocket.close()
        active_connections.remove(websocket)
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
        sample_token = decode(y[0][sample_index])
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        if sample_token != 0:
            await websocket.send(tokenizer.index2word[str(sample_token)])
        else:
            await websocket.send("<EOA>")
            return
        num_tokens_generated = len(tokens_generated)

        
async def handler(websocket, path):
    # 将新连接添加到活动连接集合
    if len(active_connections) > 30:
        await websocket.send("<BUSY>")
        websocket.close()
    active_connections.add(websocket)
    try:
        while True:
            message = await websocket.recv()
            print(message)
            await calc(message, websocket=websocket)

    except websockets.exceptions.ConnectionClosedError:
        pass

    finally:
        # 连接关闭时，从活动连接集合中移除连接
        active_connections.remove(websocket)

start_server = websockets.serve(handler, "localhost", 8080)

async def broadcast(message):
    # 广播消息给所有活动的连接
    for connection in active_connections:
        await connection.send(message)

async def main():
    server = await websockets.serve(handler, "localhost", 8080)

    try:
        while True:
            await asyncio.sleep(5)
            await broadcast("<BEAT>")

    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()

# 运行主事件循环
asyncio.run(main())