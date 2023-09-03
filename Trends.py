import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for charting
from taipy.gui import Gui, notify

text = "Original text"

page = """
<|toggle|theme|>

# Getting started with Taipy GUI

My text: <|{text}|>

<|{text}|input|>

<|Analyze|button|on_action=local_callback|>

<|{dataframe}|table|number_format=%.2f|>

<|{dataframe}|chart|type=bar|x=Text|y[1]=Score Pos|y[2]=Score Neu|y[3]=Score Neg|y[4]=Overall|color[1]=green|color[2]=grey|color[3]=red|type[4]=line|>
"""

dataframe = pd.DataFrame(
    {
        "Text": ["Test", "Other", "Love"],
        "Score Pos": [1, 1, 4],
        "Score Neu": [2, 3, 1],
        "Score Neg": [1, 2, 0],
        "Overall": [0, -1, 4],
    }
)


def local_callback(state):
    notify(state, "info", f'The text is: {state.text}')
    
    temp = state.dataframe.copy()
    state.dataframe = temp.append(
        {
            "Text": state.text,
            "Score Pos": 0,
            "Score Neu": 0,
            "Score Neg": 0,
            "Overall": 0,
        },
        ignore_index=True,
    )
    state.text = ""

    # Generate and display random bar chart
    generate_bar_chart(state.dataframe)


def generate_bar_chart(dataframe):
    # Create a bar chart based on the dataframe
    plt.figure(figsize=(8, 6))
    plt.bar(dataframe["Text"], dataframe["Overall"], color="blue")
    plt.xlabel("Text")
    plt.ylabel("Overall Score")
    plt.title("Random Bar Chart")
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.tight_layout()

    # Display the chart
    plt.show()

Gui(page).run(use_reloader=True, port=8000)

