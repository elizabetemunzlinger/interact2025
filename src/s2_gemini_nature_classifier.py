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
data from the ElicitCam dataset with the Google Gemini 2.0 Flash. It reads the
dataset, classifies each gesture based on its name and description, and saves
the classification results to a CSV file. The classification is done using the
Google Gemini API, which requires an API key set in the environment variables.
"""

# Import required libraries.
import os
import re
import time

import pandas as pd

from google import genai
from google.genai import types
from google.genai.client import Client


def get_gemini_client() -> Client:
    """Get the Google Gemini client.

    Returns:
        Client: The Google Gemini client.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError(
            "API key for Google Gemini not found in environment variables."
        )

    client = genai.Client(
        api_key=api_key
    )

    return client


def classify_hand_gesture(name: str, description: str) -> str:
    """
    Classify a hand gesture into a taxonomy class using the Google Gemini API.

    Args:
        name (str): The name of the gesture.
        description (str): The description of the gesture.

    Returns:
        str: The taxonomy class of current hand gesture.
    """

    # Get the Google Gemini client.
    client = get_gemini_client()

    # Create the Google Gemini API prompt.
    system_instruction = """
You are a specialist assistant trained to classify hand gestures according to a given taxonomy.
 
### **Classification Instructions:**
1. **If multiple gestures appear**, interpret them as a **single intentional gesture**.
2. **Use all available data** to determine the correct taxonomy classification.

### **Output Constraint:**
- Your response must be **ONLY** the exactly taxonomy class name without any other additional information or punctuation, i.e., "Metaphoric", "Iconic", "Deictic", "Symbolic", or "Pantomimic".
    """

    contents = f"""
Here is the full data to be analyzed.
- The **gesture name**: {name}.
- A **full description** of the gesture(s), detailing physical aspects: {description}.

### **Classification Taxonomy:**
This gesture should be classified into one of the following categories:
Metaphoric, Iconic, Deictic, Symbolic, or Pantomimic.
- For Methaphoric consider gestures that represent abstract concepts such as blocking/unblocking view, crossing fingers/hands, zip mouth.
- For Iconic consider gestures that visually represent concrete objects or actions close align to the speech semantic such as swipping slides, palm up/down, drawing a square in the air, grabbing/releasing.
- For Deictic consider gestures that clearly point to objects, body parts or locations such as pointing to ear/eyes, pointing screen.
- For Symbolic consider gestures that have a conventional meaning independently of the speech. They don't resemble their referent such as thumbs up (ok/agree), raised hand (stop/ask to talk), finger on lips (silent), hand/finger on ears (listen).
- For Pantomimic consider gestures that imitate real actions over objects such as close courtains, close lid, turn clockwise, slide button.

### **Analysis Guidelines:**
To determine the correct taxonomy class by considering:
- **Number of hands used**
- **Main hand parts involved**
- **Hand and finger shapes**
- **Hand and finger positions**
- **Hand and finger movements**
- **References to the body or specific head parts**
- **Specially the gesture's meaning or user intention**
        """

    # Call the Google Gemini model.
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=200,
            temperature=0,
            top_p=0.1,
        ),
    )

    # Extract the response content.
    return re.sub(r'[^a-zA-Z]', '', response.text.strip().capitalize())


if __name__ == "__main__":

    # Load the original dataset.
    file_path = os.path.join(
        os.path.dirname(__file__), "dataset/elicit_cam.csv"
    )
    df_dataset = pd.read_csv(file_path)

    # Create a dataframe for the hand gesture classification.
    file_path = os.path.join(
        os.path.dirname(__file__), "data/d2_nature_classification_gemini.csv"
    )

    columns = [ f"c{i}_nature" for i in range(1, 9) ]
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
            COMMAND_COL = f"c{id_command}_nature"
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

                    # Classify the hand gestures based on nature taxonomy.
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

                # Sleep for 4 second to avoid rate limiting.
                time.sleep(4)

    # Order the DataFrame by "id_video".
    df = df.sort_values(by="id_video").reset_index(drop=True)

    # Save the ordered DataFrame to the CSV file.
    df.to_csv(file_path, index=False)
