import json
from pathlib import Path
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import pickle

# Loading the intents data
with open('intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []

# Looping through the data to separete patterns and tags
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Tokenize the sentences
# parameters to tweak later on
vocab_size = 1000     # limit the modell to the top 1000 words
embedding_dim = 16    # How complex the word vectors should be (for step 3)
max_len = 20          # maximum length of each sentence
oov_tok = "<OOV>"     # If a word is not in the vocab, replace it with this token

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# fit the tokenizer on the training sentences
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index   # mapping of words to their index in the vocab

# Convert sentences to numbers (sequences)
sequences = tokenizer.texts_to_sequences(training_sentences)

# Make them all same length by padding
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# encode the labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)

# transform the labels and make sure they are in numpy array format
training_labels_final = lbl_encoder.transform(training_labels)
training_labels_final = np.array(training_labels_final)

# verification prints
print("Success! Data is processed.")
print(f"Vocab Size: {len(word_index)}")
print(f"Example Pattern: {training_sentences[0]} -> {padded_sequences[0]}")
print(f"Example Label: {training_labels[0]} -> {training_labels_final[0]}")


# the model
model = Sequential()

# Layer 1: Embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)) # output_dim =  how detailed should vectors be?

# layer 2: averager (globalaveragepooling1d)
model.add(GlobalAveragePooling1D())

# layer 3: hidden layer
model.add(Dense(16, activation='relu'))

# layer 4: output layer
model.add(Dense(len(labels), activation='softmax'))
 
model.build(input_shape=(None, max_len))
model.summary()

# training model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

history = model.fit(padded_sequences, training_labels_final,
                    epochs=500, verbose=1)

# save model, tokenizer, and label encoder
model.save("chat_model.keras")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

print("Model, Tokenizer, and Label Encoder saved successfully!")