from collections import defaultdict
import json
import re
from tqdm import tqdm

class Tokenizer:
    def __init__(self, filter = None, splitter=None, lower=True, reserved_symbols = [], OOV = '<UNK>', pad = '<PAD>'):
        self.lower = lower
        self.filter = filter
        self.splitter = splitter
        self.word2index = {}
        self.index2word = {}
        self.reserved_symbols = reserved_symbols
        self.reserved_symbols.append(OOV)
        self.reserved_symbols.append(pad)
        self.OOV = OOV
        self.pad = pad
        for i in range(0, len(self.reserved_symbols)):
            self.word2index[self.reserved_symbols[i]] = len(self.word2index)
            self.index2word[i] = self.reserved_symbols[i]
        self.word_counts = defaultdict(int)
        self.vocab_size = len(reserved_symbols)

    def get_vocab(self):
        return self.word2index
    
    def split_word(self, sentence):
        words = []
        if self.splitter is None:
                words = re.findall(r'\w+|[^\w\s]+', sentence)
        else:
            words = self.splitter.split(sentence)
    
        return words
    
    def fit_files(self, files):
        for _,file in tqdm(enumerate(files), desc="Tokenizer fitting on text", total=len(files)):
            with open(file, "r") as f:
                text = f.read()
                words = self.split_word(text)
                
                for word in words:
                    if self.filter is not None:
                        word = word.strip(self.filter)
                        if self.lower == True:
                            word = word.lower()
                    if word == "":
                        continue
                    if word not in self.word2index:
                        index = len(self.word2index)
                        self.word2index[word] = index
                        self.index2word[index] = word
                        self.vocab_size += 1
                    self.word_counts[word] += 1


    def fit_texts(self, texts):
        for _,text in tqdm(enumerate(texts), desc="Tokenizer fitting on text", total=len(texts)):
            words = self.split_word(text)
            
            for word in words:
                if self.filter is not None:
                    word = word.strip(self.filter)
                if self.lower == True:
                    word = word.lower()
                if word == "":
                    continue
                if word not in self.word2index:
                    index = len(self.word2index)
                    self.word2index[word] = index
                    self.index2word[index] = word
                    self.vocab_size += 1
                self.word_counts[word] += 1

    def reduce_vocab(self, x):
        if x > self.vocab_size:
            raise Exception("Attempted to reduce vocab size to a number larger than current size")
        sorted_words = sorted(self.word_counts.items(), key=lambda item: item[1], reverse=True)
        top_words = sorted_words[:x]

        self.word2index = {}
        self.index2word = {}
        for i in range(0, len(self.reserved_symbols)):
            self.word2index[self.reserved_symbols[i]] = len(self.word2index)
            self.index2word[i] = self.reserved_symbols[i]
        self.word_counts = defaultdict(int)
        self.vocab_size = len(self.reserved_symbols)

        for word, count in top_words:
            if word not in self.reserved_symbols:
                index = len(self.word2index)
                self.word2index[word] = index
                self.index2word[index] = word
                self.word_counts[word] = count
                self.vocab_size += 1


    def toSequence(self, text):
        words = self.split_word(text)
        sequence = []
        for word in words:
            if self.filter is not None:
                word = word.strip(self.filter)
            if self.lower == True:
                word = word.lower()
            if word == "":
                continue
            if word in self.word2index:
                sequence.append(self.word2index[word])
            else:
                sequence.append(self.word2index[self.OOV])
        return sequence

    def toSequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.toSequence(text)
            sequences.append(sequence)
        return sequences

    def get_word_counts(self):
        return self.word_counts
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            sequence = sequence + [self.word2index[self.pad]] * (max_length - len(sequence))
        elif len(sequence) > max_length:
            sequence = sequence[:max_length]
        return sequence

    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for sequence in sequences:
            padded_sequence = self.pad_sequence(sequence, max_length)
            padded_sequences.append(padded_sequence)
        return padded_sequences
    
    def save_vocab(self, filename):
        with open(filename, 'w') as file:
            json.dump({"word2index": self.word2index, "index2word": self.index2word}, file)

    def load_vocab(self, filename):
        with open(filename, 'r') as file:
            index_json = json.load(file)
            self.word2index = index_json["word2index"]
            self.index2word = index_json["index2word"]
            self.vocab_size = len(self.word2index)