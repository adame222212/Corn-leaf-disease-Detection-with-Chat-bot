import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
import csv
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO, BytesIO 
from flask import send_file
import base64
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


app = Flask(__name__, static_url_path='/main/static')

    # File path for CSV
CSV_FILE_PATH = "predictions.csv"
tf.keras.backend.clear_session()

    # Load necessary data and models for the chatbot
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('NLP/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

    # Load necessary data and models for image classification
MODEL_PATH = "corns.h5"
MODEL = load_model(MODEL_PATH)

CLASS_NAMES = ["Common Rust", "Gray Spot", "Healthy", "Northern Blight"]
CLASS_DIFF = [
        "It is a plant disease caused by the fungus Puccinia sorghi. It is one of the most common foliar diseases affecting corn (maize) crops. The disease is characterized by the development of small, reddish-brown to orange pustules, known as uredinia, on both sides of the corn leaves.",
        "It is a foliar disease that affects corn (maize) plants. It is caused by the fungus Cercospora zeae-maydis. Gray leaf spot can result in significant yield losses if not properly managed.",
        "This is healthy, no worries",
        "It is a foliar disease of corn caused by Exserohilum turcicum, the anamorph of the ascomycete Setosphaeria turcica. With its characteristic cigar-shaped lesions, this disease can cause significant yield loss in susceptible corn hybrids."
    
    ]

def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

def get_response(intents_list, intents_json):
        if intents_list:
            tag = intents_list[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
            return result
        else:
            # Handle the case where intents_list is empty
            return "I'm sorry, I couldn't understand your message."

    # Function to read and preprocess the file as an image for image classification
def read_file_as_image(data):
        image = Image.open(BytesIO(data))
        # Resize the image to the expected shape (256x256)
        image = image.resize((256, 256))
        image = np.array(image)
        return image


def calculate_percentage(csv_data):
    # Read the CSV data into a DataFrame
    df = pd.read_csv(io.StringIO(csv_data))
    
    class_values_list = []

    # Calculate percentages for each field (Field 1 to Field 9)
    for i in range(1, 10):
        field_name = f'Field_{i}'
       
        # Filter DataFrame for the current field and remove rows where Class is 0
        filtered_df = df[df[field_name] == 1]

        # Calculate the percentage of each class in the total count
        class_percentages = filtered_df['Class'].value_counts(normalize=True).fillna(0) * 100

        # Round the percentages to 2 decimal places
        class_percentages = class_percentages.round(2)

        # Map class names to values in the percentage plot
        class_names = class_percentages.index.tolist()
        class_names_dict = {j: class_names[i] for i, j in enumerate(range(len(class_names)))}

        class_percentages_dict = class_percentages.rename(index=class_names_dict).to_dict()
        
        # Define the threshold values for severity levels
        severe_threshold = 30
        middle_threshold = 10

        # Check severity level for "Common Rust"
        common_rust_severity = ''
        common_rust_message = ''
        if 'Common Rust' in class_percentages_dict:
            percentage = class_percentages_dict['Common Rust']
            if percentage >= severe_threshold:
                common_rust_severity = 'Severe'
                common_rust_message = "Spores will infect corn and cause symptoms within 3-4 days of infection and which can rapidly lead to an epidemic of the whole hectare."
            elif percentage >= middle_threshold:
                common_rust_severity = 'Moderate'
                common_rust_message = "Within 7 to 14 days, more urediniospores are produced and new infections continue to occur while conditions are favorable."
            else:
                common_rust_severity = 'Mild'
                common_rust_message = "Do some action before it affect others within ."
        
        northern_blight_severity = ''
        northern_blight_message = ''
        if 'Northern Blight' in class_percentages_dict:
            percentage = class_percentages_dict['Northern Blight']
            if percentage >= severe_threshold:
                northern_blight_severity = 'Severe'
                northern_blight_message = "NCLB spread quickly infecting new plants every 3-5 days it can affect rapidly for all field near for 3-4 weeks."
            elif percentage >= middle_threshold:
                northern_blight_severity = 'Moderate'
                northern_blight_message = "After 2 to 3 weeks northern blight  can transfer in near field to be affect other corn plant."
            else:
                northern_blight_severity = 'Mild'
                northern_blight_message = "Transmission rate slow down, new infection for the field  can occur every 5-7 days."
        
        healthy_severity = ''
        healthy_message = ''
        if 'Healthy' in class_percentages_dict:
            percentage = class_percentages_dict['Healthy']
            if percentage >= severe_threshold:
                healthy_severity = 'Severe'
                healthy_message = "This part is most have healthy leaf. Maintain this part and do some action for what diseases are present"
            elif percentage >= middle_threshold:
                healthy_severity = 'Moderate'
                healthy_message = "Diseases are now affecting healthy leaf. 3-7 days can be transfer, and show it will show symptoms"
            else:
                healthy_severity = 'Mild'
                healthy_message = "Low appearance of healthy leaf in this field. Monitor diseases that can affect all leafs"
        
        gray_spot_severity = ''
        gray_spot_message = ''
        if 'Gray Spot' in class_percentages_dict:
            percentage = class_percentages_dict['Gray Spot']
            if percentage >= severe_threshold:
                gray_spot_severity = 'Severe'
                gray_spot_message = "It can be significantly faster to affect a hectare, with infection possibly occurring as frequently as 3-5 times a day, leading to new plant getting infected every 2-3 days"
            elif percentage >= middle_threshold:
                gray_spot_severity = 'Moderate'
                gray_spot_message = "Within 7 to 14 days it will affect near field. Expected gray spot to spread within a field at a rate of one to two infected plants per day"
            else:
                gray_spot_severity = 'Mild'
                gray_spot_message = "Expected one infected plant per 14-21 days to transmit near corn plant"
        equal_percentage_message = ''
        equal_classes = [class_name for class_name in class_percentages_dict if all(
            class_percentages_dict[class_name] == class_percentages_dict[other_class]
            for other_class in class_percentages_dict if other_class != class_name
        )]
        if equal_classes:
            equal_percentage_message = f" {' and '.join(equal_classes)}, monitor these detected and do some some action"

        class_values_list.append({
            'field_name': field_name,
            'class_percentages': class_percentages_dict,
            'common_rust_severity': common_rust_severity,
            'common_rust_message': common_rust_message,
            'northern_blight_severity': northern_blight_severity,
            'northern_blight_message': northern_blight_message,
            'healthy_severity': healthy_severity,
            'healthy_message': healthy_message,
            'gray_spot_severity': gray_spot_severity,
            'gray_spot_message': gray_spot_message,
            'equal_percentage_message': equal_percentage_message
        })

    return class_values_list



def generate_combined_count_plot(csv_data):
    # Read the CSV data into a DataFrame
    df = pd.read_csv(io.StringIO(csv_data))

    # Generate individual count plots for each field
    plt.figure(figsize=(15, 10))

    for i in range(1, 10):
        field_name = f'Field_{i}'

        # Filter DataFrame for the current field and remove rows where Class is 0
        filtered_df = df[df[field_name] == 1]

        # Create a count plot for the current field
        plt.subplot(3, 3, i)
        sns.countplot(data=filtered_df, x='Class')
        plt.title(f'Field {i}')
        plt.xlabel('Class')
        plt.ylabel('Number of diseases')

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Clear the plot to avoid issues with subsequent requests
    plt.clf()
    plt.close()

    # Return the base64-encoded image
    return f"data:image/png;base64,{base64.b64encode(img_buf.getvalue()).decode('utf-8')}"


def average_classes(csv_file_path, class_column):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the specified class column exists in the DataFrame
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in the CSV file.")

    # Calculate the average occurrences of each class
    total_rows = len(df)
    class_counts = df[class_column].value_counts()
    class_avg = class_counts / total_rows

    # Convert class averages to a dictionary
    class_avg_dict = class_avg.to_dict()

    return class_avg_dict


@app.route('/')
def home():
        return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
        user_message = request.form['user_message']
        ints = predict_class(user_message)
        res = get_response(ints, intents)
        return res

@app.route('/image_classification', methods=['POST'])
def image_classification():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Read the selected value from the dropdown
            selected_value = int(request.form.get('button_value'))

            image = read_file_as_image(file.read())
            img_batch = np.expand_dims(image, 0)

            predictions = MODEL.predict(img_batch)

            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            predicted_DIF = CLASS_DIFF[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            confidence = round(confidence, 2)
            # Save the predictions and selected value to a CSV file with a header row
            with open(CSV_FILE_PATH, mode='a', newline='') as csvfile:
                # Update the 'Confidence ' to 'Confidence'
                fieldnames = ['Class'] + [f'Field_{i}' for i in range(1, 10)] + ['Selected_Field', 'Confidence']

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header row if the file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()

                # Create a dictionary with 0 for all fields
                row_data = {'Class': predicted_class, 'Selected_Field': selected_value, 'Confidence': confidence}
                for i in range(1, 10):
                    row_data[f'Field_{i}'] = 1 if selected_value == i else 0

                writer.writerow(row_data)

            return jsonify({
                'class': predicted_class,
                'Diff': predicted_DIF
            })

@app.route('/combined_visualization', methods=['POST'])
def combined_visualization():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    csv_data = file.read().decode('utf-8')

    # Generate the class values for percentage plots
    class_values_list = calculate_percentage(csv_data)

    # Generate the combined count plot for all fields
    img_src_combined_count_plot = generate_combined_count_plot(csv_data)

    # Call the average_classes function instead of count_classes
    average_classes_result = average_classes(StringIO(csv_data), "Class")

    return render_template('combined_visualization.html', class_values_list=class_values_list, img_src_combined_count_plot=img_src_combined_count_plot, average_classes_result=average_classes_result)

@app.route('/area_analysis')
def area_analysis():
     return render_template('area_analysis.html')

if __name__ == '__main__':
        app.run(debug=True)