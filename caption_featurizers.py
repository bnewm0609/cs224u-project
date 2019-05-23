import nltk
from collections import Counter
import torch
import numpy as np



class Tokenizer:
    def tokenize(self, sentence):
        pass

class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)
    
class EndingTokenizer(Tokenizer):
    """
    Segments endings as different words from words that end with them:
    Ex: 'greener' -> 'green', 'er'
    """
    def __init__(self):
        # Endings defined here:
        # https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/tokenizers.py#L35
        self.endings = ['er', 'est', 'ish']
        
    def tokenize(self, sentence):
        tokens = []
        for word in nltk.word_tokenize(sentence):
            inserted = False
            for ending in self.endings:
                if word.endswith(ending):
                    tokens.extend([word[:-len(ending)], '+{}'.format(ending)])
                    inserted = True
                    break
            if not inserted:
                tokens.append(word)
        return tokens


class CaptionIndexer:
    def __init__(self):
        self.UNK = '<unk>'
        self.EOS = '<eos>'
        self.SOS = '<sos>'
        
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.size = 0
        
        
    def add_sentence(self, sentence):
        for word in sentence:
            word = word.lower()
            if not word in self.word2idx:
                self.word2idx[word] = self.size
                self.idx2word[self.size] = word
                self.size += 1
            self.word_count[word] += 1
        
    def get_word_from_idx(self, idx):
        return self.idx2word[idx]
    
    def get_idx_from_word(self, word):
        return self.word2idx.get(word, self.word2idx[self.UNK])
    
    def to_indices(self, sentence, construct=False):
        if construct:
            self.add_sentence(sentence)
            # we know everything is in the map because we just added it
            return [self.word2idx[word] for word in sentence]
        
        return [self.get_idx_from_word(word) for word in sentence]


class CaptionFeaturizer:
    
    def __init__(self, tokenizer=WhitespaceTokenizer, unk_threshold=1):
        self.tokenizer = tokenizer()
        self.caption_indexer = CaptionIndexer()
        self.word_count = None
        
        # hyperparams
        self.unk_threshold = unk_threshold

        self.initialized = False

    
    def to_features(self, data_entry):
        return self.to_string_features(data_entry.caption)

    def to_tensor(self, caption, construct=False):
        _, indexes = self.to_string_features(caption, construct)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)
    
    def to_string_features(self, caption, construct=False):
        """
        Params:
        caption:   string hodling caption that will converted to tokens and
                   indices. 
        construct: if we are constructing the featurizer for the first time,
                   this should be true. It performs the unk substitutions 
                   manually based on the contents of self.word_count and 
                   also adds the sentences to the indexer. Should only be
                   true when training for the first time.
                   
        Returns:
        Tuple(tokens, indices). 
        
        tokens is a tokenized version the passed caption,
                unks are replaced, words are lower cased, buffered on both sides by sos/
                eos tags
        indices is a list indices given by the indexer for each token. These can be converted
                to tensor to be fed into the model
        
        """
        construct = construct and not self.initialized # don't construct index if we already did in construct_featurizer
        # automatically construct the index if it isn't initialized yet
        caption_tokens = self.tokenizer.tokenize(caption)
        caption_tokens = self.to_model_format(caption_tokens, construct)
        caption_indices = self.caption_indexer.to_indices(caption_tokens, construct)
        caption_tokens = [self.caption_indexer.get_word_from_idx(index) for index in caption_indices]


        return np.array(caption_tokens), np.array(caption_indices)
    
    def to_model_format(self, tokens, construct):
        """
        Put the tokens into the format expected by the models.
        This mainly entails prepending/appending <sos>, <eos>,
        lowercasing all of the words and replacing all uncommon words
        with <unk> (only in the case when we are constructing the
        featurizer for the first time)
        
        Params:
        tokens: 
        construct: if we are constructing the featurizer for the first time,
                   this should be true. It performs the unk substitutions 
                   manually based on the contents of self.word_count and 
                   also adds the sentences to the indexer. Should only be
                   true when training for the first time.
        """
        if construct:
            if self.word_count is None:
                print("FEATURIZER HAS NOT BEEN CONSTRUCTED YET. Call `construct_featurizer`")
            else:
                for i in range(len(tokens)):
                    if self.word_count[tokens[i]] <= self.unk_threshold:
                        tokens[i] = self.caption_indexer.UNK
                        
        tokens = [token.lower() for token in tokens]
        tokens = [self.caption_indexer.SOS] + tokens + [self.caption_indexer.EOS]
        return tokens
    
    def construct_featurizer(self, data_entries, construct_idx=True):
        """
        data_entries is of type MonroeData. 
        """
        self.word_count = Counter()
        for entry in data_entries:
            caption_tokens = self.tokenizer.tokenize(entry.caption)
            for token in caption_tokens:
                self.word_count[token] += 1

        if construct_idx:
            # just construct the index so we don't have to worry about calling
            # anything with the construct=True argument to to_string_features
            for entry in data_entries:
                _ = self.to_string_features(entry.caption, construct=True)
            self.initialized = True





