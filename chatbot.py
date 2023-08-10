
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:\\Users\\Supreme Kandel\\Desktop\\Chatbot\\intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')

def preprocess_input(text):
    text = text.lower()
    return nltk.word_tokenize(text)

# def get_response(model, classes, input_text):
#     input_words = preprocess_input(input_text)
#     input_bag = [1 if word in input_words else 0 for word in words]
#     predictions = model.predict(np.array([input_bag]))[0]
#     predicted_class_index = np.argmax(predictions)
#     response_tag = classes[predicted_class_index]
#     return random.choice([response['response'] for response in intents['intents'] if response['tag'] == response_tag]['responses'])
def get_response(model, classes, input_text):
    input_words = preprocess_input(input_text)
    input_bag = [1 if word in input_words else 0 for word in words]
    predictions = model.predict(np.array([input_bag]))[0]
    predicted_class_index = np.argmax(predictions)
    response_tag = classes[predicted_class_index]
    
    matching_intent = [intent for intent in intents['intents'] if intent['tag'] == response_tag]
    if matching_intent:
        selected_intent = matching_intent[0]
        if 'responses' in selected_intent and len(selected_intent['responses']) > 0:
            return random.choice(selected_intent['responses'])
        else:
            return "I'm not sure how to respond to that."
    else:
        return "I'm not sure how to respond to that."

# Rest of the code...


print("Chatbot: Hi there! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = get_response(model, classes, user_input)
    print("Chatbot:", response)

print('Done')
