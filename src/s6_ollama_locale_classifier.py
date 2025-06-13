#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2025 Elizabete Munzlinger, Fabricio Batista Narcizo, Renata
# Briet, Mario Tadashi Shimanuki, Ted Vucurevich, and Dan Witzner Hansen.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""This Python script classifies a hand gesture into a taxonomy class using
data from the ElicitCam dataset with the llama3.3. It reads the dataset,
classifies each gesture based on its name and description, and saves the
classification results to a CSV file. The classification is done using the
Ollama API with local LLM models."""

# Import required libraries.
import os
import re

import pandas as pd

from ollama import Client


def get_ollama_client() -> Client:
    """Get the Ollama client.

    Returns:
        Client: The Ollama client.
    """
    client = Client(
        host="http://localhost:11434"
    )

    return client


def classify_hand_gesture(name: str, description: str) -> str:
    """
    Classify a hand gesture into a taxonomy class using Ollama.

    Args:
        name (str): The name of the gesture.
        description (str): The description of the gesture.

    Returns:
        str: The taxonomy class of current hand gesture.
    """

    # Get the Ollama client.
    client = get_ollama_client()

    # Create the Ollama prompt.
    system_prompt = """
You are a specialist assistant trained to classify hand gestures according to a given taxonomy.
 
### **Classification Instructions:**
1. **If multiple gestures appear**, interpret them as a **single intentional gesture**.
2. **Use all available data** to determine the correct taxonomy classification.

### **Output Constraint:**
- Your response must be **ONLY** the exactly taxonomy class name without any other additional information or punctuation, i.e., "Mouth Interaction", "Ear Interaction", "Eyes Interaction", "Other Self Interaction", or "No Self Interaction".
    """

    user_prompt = f"""
Here is the full data to be analyzed.
- The **gesture name**: {name}.
- A **full description** of the gesture(s), detailing physical aspects: {description}.

### **Classification Taxonomy:**
This gesture should be classified into one of the following categories:
Mouth Interaction, Ear Interaction, Eyes Interaction, Other Self Interaction, No Self Interaction.
- For Mouth Interaction, consider the gesture which refers specifically to the mouth or lips in any kind of interaction.
- For Ear Interaction, consider the gesture which which refers specifically to the ear in any kind of interaction.
- For Eye Interaction, consider the gesture which refers specifically to the eye/eyes (not face) in any kind of interaction.
- For Other Self Interaction, consider the gesture which refers specifically to any other specific part of the head or body such as face, chin, chest, head, in any kind of interaction.
- For No Self Interaction, consider the gesture which doesn't refer to any part of the head or body.

### **Analysis Guidelines:**
To determine the correct taxonomy class by considering:
- **The presence of the referred head part on the name or description**
        """

    # Prepare the messages for the model.
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    # Call the model.
    response = client.chat(
        model="llama3.3",
        messages=messages,
        options={
            "temperature": 0,
            "top_p": 0.1,
        }
    )

    # Extract the response content.
    response_content = response["message"]["content"]

    # For DeepSeek model, remove content between <think> and </think> tags to
    # remove thinking output.
    # response_content = \
    #     re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL)

    return re.sub(r'[^a-zA-Z] ', '', response_content.strip())


if __name__ == "__main__":

    # Load the original dataset.
    file_path = os.path.join(
        os.path.dirname(__file__), "dataset/elicit_cam.csv"
    )
    df_dataset = pd.read_csv(file_path)

    # Create a dataframe for the hand gesture classification.
    file_path = os.path.join(
        os.path.dirname(__file__), "data/d6_locale_classification_llama.csv"
    )

    columns = [ f"c{i}_locale" for i in range(1, 9) ]
    columns.insert(0, "id_video")  # Add "id_video" as the first column.

    # Load the existing annotations or create a new DataFrame.
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns).astype(str)

    # Iterate over each row in the dataset.
    for index, row in df_dataset.iterrows():

        # Ensure the id_video exists in the DataFrame.
        id_video = row["id_video"]
        if id_video not in df["id_video"].values:
            df = pd.concat([
                df, pd.DataFrame({ "id_video": [id_video] })
            ], ignore_index=True)

        # Iterate over each command in the row.
        for id_command in range(1, 9):
            COMMAND_COL = f"c{id_command}_locale"
            print(f"Processing {id_video} / command {id_command}...")

            # Check if the data has not already been annotated.
            if (df.loc[df["id_video"] == id_video, COMMAND_COL].isna().all()):
                try:

                    # Select the data annotations for the current data.
                    command_cols = [
                        col for col in df_dataset.columns
                        if col.startswith(f"c{id_command}")
                    ]

                    annotations = row[command_cols]

                    # Classify the hand gestures based on locale taxonomy.
                    result = classify_hand_gesture(
                        annotations.values[2],  # Gesture's name
                        annotations.values[3],  # Gesture's description
                    )

                # Handle the case where the content is blocked.
                except RuntimeError as e:
                    print(f"Exception Error: {str(e)}.")
                    continue

                # Add the result to the corresponding row and column in the
                # DataFrame.
                df.loc[df["id_video"] == id_video, COMMAND_COL] = result

                # Save the updated DataFrame to the CSV file.
                df.to_csv(file_path, index=False)

    # Order the DataFrame by "id_video".
    df = df.sort_values(by="id_video").reset_index(drop=True)

    # Save the ordered DataFrame to the CSV file.
    df.to_csv(file_path, index=False)
