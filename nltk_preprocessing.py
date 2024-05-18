import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# Tokenization
def word_tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stemming and Converting to  Lowercase 
stemmer=PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower()) 

# Bag of Words
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
    
'''
a="WHat's is the college address ?"
print(word_tokenize(a))
print(stem(a))
sentence = ["hello", "how", "are", "you"]
words = ["hiiiiiii", "hello", "IMJ", "you", "byeee", "thank", "coolll"]
b=bag_of_words(sentence,words)
print(b)
'''