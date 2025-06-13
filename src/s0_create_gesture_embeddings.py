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

"""This Python script creates embeddings for each hand gestures in the dataset
and saves them to a CSV file. The embeddings are generated using a pre-trained 
transformer model (DeBERTa) from the Hugging Face Transformers library."""

# Import required libraries.
from enum import Enum

import os
import torch

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from transformers.models.deberta.modeling_deberta import DebertaModel
from transformers.models.deberta.tokenization_deberta import DebertaTokenizer


class Gesture(Enum):
    """Enum class to represent the different hand gestures."""
    C1_COMMAND = "c1_command"
    C2_COMMAND = "c2_command"
    C3_COMMAND = "c3_command"
    C4_COMMAND = "c4_command"
    C5_COMMAND = "c5_command"
    C6_COMMAND = "c6_command"
    C7_COMMAND = "c7_command"
    C8_COMMAND = "c8_command"


def get_embedding(
        deberta_model: DebertaModel,
        deberta_tokenizer: DebertaTokenizer,
        gesture: str
    ) -> np.ndarray:
    """
    Get the embedding of a gesture text description using a pre-trained model.

    Args:
        deberta_model: The pre-trained model.
        deberta_tokenizer: The tokenizer used to tokenize the text.
        gesture (str): The text description of the gesture.

    Returns:
        np.ndarray: The embedding of the gesture.
    """

    # Tokenize the gesture description and get the embeddings.
    inputs = deberta_tokenizer(
        gesture, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        outputs = deberta_model(**inputs)

    # Get the embedding of the [CLS] token.
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


if __name__ == "__main__":

    # Define the file path and column names.
    file_path = os.path.join(
        os.path.dirname(__file__), "dataset/elicit_cam.csv"
    )
    columns = [ f"c{i}_description" for i in range(1, 9) ]
    columns.insert(0, "id_video")  # Add "id_video" as the first column.

    # Load the original dataset.
    df = pd.read_csv(file_path)

    # Create a data frame with the hand gesture commands.
    df_gestures = df[columns]

    # Melt the data frame to have the description in a single column.
    df_gestures = df_gestures.melt(
        id_vars="id_video", var_name="id_description", value_name="description"
    )

    # Load the pre-trained model and tokenizer.
    MODEL_NAME = "microsoft/deberta-xlarge-mnli"
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create embeddings for each gesture.
    df_gestures["embedding"] = df_gestures["description"].apply(
        lambda gesture: get_embedding(model, tokenizer, gesture).tolist()
    )

    # Save the ordered DataFrame to the CSV file.
    file_path = os.path.join(
        os.path.dirname(__file__), "data/d0_hand_gesture_descriptions.csv"
    )
    df_gestures.to_csv(file_path, index=False)
