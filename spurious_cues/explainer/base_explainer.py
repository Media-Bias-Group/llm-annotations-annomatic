from abc import ABC, abstractmethod
from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pickle
from tqdm import tqdm
from spurious_cues.myutils import clean_token_text
import numpy as np
import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image


class BaseExplainer(ABC):
    """Abstract class for explainers."""

    def __init__(
        self,
        model_checkpoint: str,
        dataset: str,
        top_k=5,
        split='train',
        class_names=["unbiased", "biased"],
    ):
        """
        Initializes the Explainer object.

        Args:
            model_checkpoint (str): The path to the model checkpoint file.
            dataset (str): The name of the dataset.
            top_k (int, optional): The number of top predictions to consider. Defaults to 5.
        """
        self.model_checkpoint = model_checkpoint
        self.split = split
        self.dataset = load_dataset(dataset)[self.split].to_pandas()[:10]
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
        )
        self.top_k = top_k
        self.class_names = class_names
        self.token2att = None
        self.counter = None
        self.last_explained_class = None
        self.model.eval()

    @abstractmethod
    def explain_sentence(self, sent) -> Dict[str, float]:
        """
        Explains the given sentence and returns a dictionary of most important words.

        Args:
            sent (str): The sentence to be explained.

        Returns:
            Dict[str, float]: A dictionary containing explanation scores for different tokens of the sentence.
        """

    def explain_dataset(self, class_=1, path: str = None):
        """
        Explain the dataset for the specified class.

        Args:
            class_ (int): The class for which to provide explanations. Defaults to 1.
            path: Path to load the explanations from. Defaults to None.

        Returns:
            None
        """
        class_str = self.class_names[class_]
        self.last_explained_class = class_

        # load from file if exists
        if path is not None:
            if os.path.exists(path):
                with open(
                    os.path.join(path, f"token2att_{class_str}.pkl"),
                    "rb",
                ) as f:
                    self.token2att = pickle.load(f)
                with open(
                    os.path.join(path, f"counter_{class_str}.pkl"),
                    "rb",
                ) as f:
                    self.counter = pickle.load(f)
                return

        token2att = {}
        counter = {}

        data = self.dataset[self.dataset.label == class_]["text"].tolist()

        for text in tqdm(data):
            att_dict = self.explain_sentence(text)

            for tok, score in att_dict.items():
                tok = clean_token_text(tok)
                # add to mapping
                if tok in token2att.keys():
                    token2att[tok] += score
                elif tok not in token2att.keys():
                    token2att[tok] = score

                # add to counter
                if tok in counter.keys():
                    counter[tok] += 1
                elif tok not in counter.keys():
                    counter[tok] = 1

        # average the attention scores for each token
        df = pd.DataFrame({'attribution':token2att,'count':counter})
        df['attribution'] = df['attribution']/df['count']
        df.to_csv('token_attributions.csv')

    def create_wordcloud_text(self):
        """Create long string of text from the list of tokens by weighting them by their number
        of occurances and the average attention weight."""
        tokens = list(self.token2att.keys())

        token_str_list = []

        for token in tokens:
            # final_count = int(self.token2att[token]*self.counter[token])
            final_count = int(self.token2att[token] * 1000)
            token_str_list.extend([token] * final_count)

        # shuffle prevents words from being grouped together by wordcloud
        np.random.shuffle(token_str_list)
        return " ".join(token_str_list)

    def get_wordcloud(self, silhouette_path=None):
        """Generate wordcloud from the list of tokens."""
        mask = (
            np.array(Image.open(silhouette_path)) if silhouette_path else None
        )
        color = "Greens" if self.last_explained_class == 0 else "Reds"
        wc = WordCloud(
            width=1600,
            height=800,
            background_color="white",
            mask=mask,
            colormap=color,
        )
        wordcloud = wc.generate(self.create_wordcloud_text())

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # Hide the axes
        class_str = self.class_names[self.last_explained_class]
        plt.savefig(
            f"data/figures/{class_str}.png",
            dpi=500,
            bbox_inches="tight",
        )
