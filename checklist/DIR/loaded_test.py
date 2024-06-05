from checklist import BaseTest
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
)
from datasets import load_dataset
import spacy
import random
from tqdm import tqdm

BIASED = 1
UNBIASED = 0


class LoadedTest(BaseTest):
    def __init__(self, data_path, k=10):
        super().__init__(data_path)
        self.pos_pipe = spacy.load("en_core_web_sm")
        self.k = k

    def inject_biased_adjective(self, text: str, adjectives):
        """
        Injects a biased adjective into a given text by modifying a suitable noun or noun phrase.

        Args:
            text (str): The input text to modify.
            adjectives (list): A list of biased adjectives to choose from.

        Returns:
            str: The modified sentence with the biased adjective injected.
        """

        text_ = self.pos_pipe(text)
        new_tokens = []
        injected = False

        for token in text_:
            # Append the token text to the new token list
            new_tokens.append(token.text)
            # Find a suitable noun or noun phrase to modify
            if (
                token.pos_ == "NOUN"
                and not injected
                and token.dep_ in ["nsubj", "dobj", "attr"]
            ):
                adj = random.choice(adjectives)
                # Check if there's an adjective before the noun
                if token.n_lefts > 0 and next(token.lefts).pos_ == "ADJ":
                    # Place the biased adjective after the existing adjective
                    new_tokens.insert(len(new_tokens) - 1, adj)
                else:
                    # Place the biased adjective before the noun
                    new_tokens.insert(len(new_tokens) - 1, adj)
                # injected = True

        # Join the tokens to form the modified sentence
        modified_sentence = " ".join(new_tokens)

        return modified_sentence

    def inject_biased_adverb(self, text: str, adverbs):
        """
        Injects a biased adverb into a given sentence after a suitable verb.

        Args:
            text (str): The input sentence.
            adverbs (list): A list of biased adverbs to choose from.

        Returns:
            str: The modified sentence with the biased adverb injected.

        """
        text_ = self.pos_pipe(text)
        # Create a list to hold the modified sentence tokens
        new_tokens = []
        injected = False

        for token in text_:
            # Append the token text to the new token list
            new_tokens.append(token.text)
            # Find a suitable verb to modify
            if token.pos_ == "VERB" and not injected:
                if token.dep_ in [
                    "ROOT",
                    "aux",
                    "xcomp",
                    "ccomp",
                    "advcl",
                ] or (token.head.pos_ == "VERB"):
                    adv = random.choice(adverbs)
                    # Insert the biased adverb immediately after the verb
                    new_tokens.append(adv)
                    # injected = True

        # Join the tokens to form the modified sentence
        modified_sentence = " ".join(new_tokens)

        return modified_sentence

    def prepare_test_data(self):
        """
        Prepares the test data by retrieving tables from the 'minorities.db' database
        and storing them in a dictionary.
        """

        lexicon = load_dataset("mediabiasgroup/bias-lexicon")[
            "train"
        ].to_pandas()
        df = load_dataset("mediabiasgroup/BABE")["test"].to_pandas()

        adjectives = lexicon[lexicon.pos == "ADJ"]["word"].tolist()
        adverbs = lexicon[lexicon.pos == "ADV"]["word"].tolist()
        df = df[df.label == 0]  # only neutral
        print("Injecting biased adjectives and adverbs...")

        texts_orig = []
        texts_loaded = []
        labels = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            orig_text = row["text"]
            new_texts_adj = [
                self.inject_biased_adjective(orig_text, adjectives)
                for _ in range(self.k)
            ]
            new_texts_adv = [
                self.inject_biased_adverb(orig_text, adverbs)
                for _ in range(self.k)
            ]

            texts_orig.extend([orig_text] * (self.k * 2))
            labels.extend([UNBIASED] * (self.k * 2))
            texts_loaded.extend(new_texts_adj + new_texts_adv)

        self.test_data = pd.DataFrame.from_dict(
            {
                "text_orig": texts_orig,
                "text_loaded": texts_loaded,
                "label": labels,
            }
        )

    def test(self, test_data):
        # first only evaluate on original test data, and keep only the instances
        # where model is correct
        orig_data = test_data[~test_data.text_orig.duplicated()]
        orig_data["preds_orig"] = self.make_predictions(
            target_col="text_orig", data=orig_data
        )
        orig_data = orig_data[orig_data.preds_orig == orig_data.label]
        # out of full test_data filter out instances where model was correct
        # in the first place
        test_data = orig_data[["text_orig"]].merge(test_data, on="text_orig")

        # we added the loaded words, therefore all labels are expected to change to BIASED
        test_data["label"] = [BIASED] * len(test_data["label"])
        test_data["preds"] = self.make_predictions(target_col="text_loaded",data=test_data)

        return test_data
