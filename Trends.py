from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import random


import numpy as np
import pandas as pd 
from taipy.gui import Gui, notify

text = "Original text"

page = """
# IoT Weather Analysis App

![physical_iot](https://github.com/Nishu0/WeatherAnalysis/assets/89217455/2ed674c8-0ac3-4bf9-8190-c4a3bcac04c6)

## Welcome to IoT Weather Analysis

Welcome to our IoT Weather Analysis App, a powerful tool for exploring weather trends and conducting in-depth analysis of weather data. With this app, you can gain valuable insights into weather conditions using data gathered from IoT devices such as ESP32, DHT11, and BMP180.

### What It Does

Our IoT Weather Analysis App allows you to:

- Monitor Real-time Weather Data: Get access to real-time weather data collected by IoT devices placed in different locations.

- Analyze Historical Trends: Explore historical weather trends and patterns to make informed decisions.

- Visualize Data: Visualize weather data using interactive charts and graphs, making it easier to understand complex weather patterns.

- Predict Future Weather: Utilize data analysis to predict future weather conditions, helping you plan your activities accordingly.

![download](https://github.com/Nishu0/WeatherAnalysis/assets/89217455/26a28b2e-a8cb-42a9-9d79-f830bd4fd47f )

### How It Works

Our app relies on the following IoT devices for data gathering:

- **ESP32:** This versatile microcontroller is used to collect data from various weather sensors.

- **DHT11:** The DHT11 sensor measures temperature and humidity, providing crucial climate information.

- **BMP180:** This barometric pressure sensor helps us gauge atmospheric pressure changes, which can indicate weather shifts.

The IoT devices continuously collect data and transmit it to our app, where it is processed and analyzed using advanced algorithms and machine learning models.

![flowchart](https://github.com/Nishu0/WeatherAnalysis/assets/89217455/0028ec5a-025d-4856-b282-514926ce9d6c)

### Key Features

- **Real-time Weather Data:** Access up-to-the-minute weather information from multiple sensors.

- **Historical Analysis:** Dive into historical weather data to uncover trends and patterns.

- **Interactive Visualizations:** Visualize weather data with interactive charts and graphs for better insights.

- **Weather Predictions:** Use data-driven predictions to plan for future weather conditions.

- **User-Friendly Interface:** Our user-friendly interface makes it easy for anyone to explore and understand weather data.

### What We Used

Our IoT Weather Analysis App is built using Python, Taipy GUI, and popular machine learning models for accurate data analysis.

### Get Started

To get started, simply enter a location or select a sensor from the dropdown menu to view weather data. You can also explore historical data, analyze trends, and make data-driven decisions for your weather-related activities.

Discover the power of IoT-driven weather analysis with our app. Start exploring weather trends and making informed decisions today!

---

#### Connect with Us

Follow us on social media for updates and news:

- [Twitter](#)
- [LinkedIn](#)

#### Contact Us

If you have any questions or feedback, feel free to reach out to us at [itsnisargthakkar@email.com](mailto:itsnisargthakkar@email.com).

#### Privacy Policy

Read our [Privacy Policy](#) to learn about how we handle your data and ensure your privacy.

"""


about_page = """
# About

This project was created by Nisarg and Pratham.


<|Email|button|on_action=local_callback|>


<|LinkedIn|button|on_action=local_callback|>


<|Github|button|on_action=local_callback|>




  Made with :coffee: and ❤️ for MLH Hackathon Hack the Classroom.


"""

form_page = """
# Weather Data Input Form

Temperature: <|{temp}|input|>


Pressure: <|{pres}|input|>


Humidity: <|{humd}|input|>


Wind Speed: <|{wind}|input|>
<br />



Rating: <|{value}|slider|min=1|max=10|>
<br />


Want Data: <|{value}|toggle|lov=Yes;No|>


 <|button|label=Submit|>





"""


def get_data(path_to_csv: str):
    # pandas.read_csv() returns a pd.DataFrame
    dataset = pd.read_csv(path_to_csv)
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    return dataset


# Read the dataframe
path_to_csv = "dataset.csv"
dataset = get_data(path_to_csv)

# Initial value
n_week = 10

trend_page = """
# Weather Data Trends

Week number: *<|{n_week}|>*

Interact with this slider to change the week number:

<|{n_week}|slider|min=1|max=52|>

## Dataset:

Display the last three months of data:
<|{dataset[9000:]}|chart|type=bar|x=Date|y=Value|>

<br/>

<|{dataset}|table|width=100%|>



"""


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

dataframe = pd.DataFrame({"Text":[''],
                          "Temp":[0.33],
                          "Pressure":[0.33],
                          "Humidity":[0.33],
                          "WindSpeed":[0]})

dataframe2 = dataframe.copy()

def analyze_text(text):
    # Run for Roberta Model
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    return {"Text":text[:50],
            "Score Pos":scores[2],
            "Score Neu":scores[1],
            "Score Neg":scores[0],
            "Overall":scores[2]-scores[0]}


def local_callback(state):
    notify(state, 'Info', f'The text is: {state.text}', True)
    temp = state.dataframe.copy()
    scores = analyze_text(state.text)
    state.dataframe = temp.append(scores, ignore_index=True)
    state.text = ""


path = ""
treatment = 0

page_file = """
<|toggle|theme|>

# Weather Data Analysis

<|{dataframe}|table|number_format=%.2f|>

<|{dataframe}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|>
"""

dataframe = pd.DataFrame({"Text":['Temperature', 'Humudity', 'Pressure'],
                          "Score Pos":[1, 1, 4],
                          "Score Neu":[2, 3, 1],
                          "Score Neg":[1, 2, 0],
                          "Overall":[0, -1, 4]})


def local_callback(state):
    notify(state, 'info', f'The text is: {state.text}')
    
    temp = state.dataframe.copy()
    state.dataframe = temp.append({"Text":state.text,
                                   "Score Pos":0,
                                   "Score Neu":0,
                                   "Score Neg":0,
                                   "Overall":0}, ignore_index=True)
    state.text = ""

def analyze_file(state):
    state.dataframe2 = dataframe2
    state.treatment = 0
    with open(state.path,"r", encoding='utf-8') as f:
        data = f.read()
        
        # split lines and eliminates duplicates
        file_list = list(dict.fromkeys(data.replace('\n', ' ').split(".")[:-1]))
    
    
    for i in range(len(file_list)):
        text = file_list[i]
        state.treatment = int((i+1)*100/len(file_list))
        temp = state.dataframe2.copy()
        scores = analyze_text(text)
        state.dataframe2 = temp.append(scores, ignore_index=True)
        
    state.path = None
    

pages = {"/":"<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
         "Home":page,
         "Analysis":page_file,
         "About": about_page,
         "Form": form_page,
         "Trends": trend_page}

Gui(pages=pages).run(use_reloader=True, port=8000)


