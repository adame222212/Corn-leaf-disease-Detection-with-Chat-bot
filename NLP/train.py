# Importing necessary libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Initializing WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Loading intents from JSON
with open('NLP/intents.json') as file:
    intents = json.load(file)

# Preprocessing intents and tokenizing words
words, classes, documents = [], [], []

# Defining characters to ignore
ignore_letters = ['!', '&', '?', '.', ',']

# Tokenizing patterns and building word list
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words.extend(nltk.word_tokenize(pattern))
        documents.append((nltk.word_tokenize(pattern), intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatizing and removing ignored characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Saving words and classes to pickle files
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Initializing training data
training = []
output_empty = [0] * len(classes)

# Creating bag of words for each pattern
for document in documents:
    bag = [1 if word in document[0] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffling training data
random.shuffle(training)
training = np.array(training)

# Separating features (X) and labels (y)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Creating a Sequential model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compiling model with Adam optimizer
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Training the model
model.fit(train_x, train_y, epochs=1000, batch_size=100, verbose=1)

# Saving the model
model.save('chatbotmodel.h5')

# Evaluating the model
predictions = model.predict(train_x)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(train_y, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(true_classes, predicted_classes, target_names=classes)
print("Classification Report:")
print(class_report)

# Plotting Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
