import json
import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import pickle

# ---------------------------------------------------------
# SETUP: LOAD EVERYTHING
# ---------------------------------------------------------
print("Loading the brain...")

# 1. Load the intents (for the text responses)
with open('intents.json') as file:
    data = json.load(file)

# 2. Load the trained model
model = load_model('chat_model.keras')

# 3. Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 4. Load the label encoder
with open('label_encoder.pickle', 'rb') as enc_file:
    lbl_encoder = pickle.load(enc_file)

# PARAMETERS (MUST MATCH TRAINING!)
max_len = 20


# THE CHAT LOOP

print("Bot is ready! (Type 'quit' to stop)")

while True:
    # 1. Get User Input
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # 2. Preprocessing (Exact same steps as training)
    # Note: texts_to_sequences expects a LIST, so we wrap user_input in []
    seq = tokenizer.texts_to_sequences([user_input]) 
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)

    # 3. Prediction
    # result is a list of probabilities: [[0.02, 0.95, 0.03]]
    result = model.predict(padded, verbose=0) 
    
    # argmax gives us the index of the highest number: 1
    tag_index = np.argmax(result)
    
    # 4. Decoding (Number -> Tag Name)
    tag = lbl_encoder.inverse_transform([tag_index])[0]
    
    # 5. Confidence Check
    confidence = result[0][tag_index]

    if confidence > 0.6: # Only answer if 60% sure
        for intent in data['intents']:
            if intent['tag'] == tag:
                # Pick a random response from the list
                print(f"Bot: {np.random.choice(intent['responses'])}")
    else:
        print("Bot: I'm not sure I understand that. Can you rephrase?")