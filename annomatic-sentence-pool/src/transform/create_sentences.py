import pandas as pd
from langdetect import detect
from tqdm import tqdm
import re
from src.utils_ import to_parquet
from nltk.tokenize import sent_tokenize
import uuid

INPUT_PATH = "data/transform/tmp"
OUTPUT_PATH = "data/transform/tmp"
tqdm.pandas()


def _split_into_sentences(article_text, title):
    """
    Splits an article text into sentences.

    Args:
        article_text (str): The article text to be split.

    Returns:
        list: The list of sentences.
    """
    sentences = [title] + sent_tokenize(article_text)

    def valid_sentence(sent):
        if sent is None:
            return False
        if len(sent) <= 10:
            return False
        return True

    sentences = [
        sentence for sentence in sentences if valid_sentence(sentence)
    ]
    return sentences


@to_parquet(f"{OUTPUT_PATH}/sentences.parquet")
def main():
    articles_df = pd.read_parquet(f"{INPUT_PATH}/articles.parquet")

    # split into sentences
    sentence_dataframes = []
    for _, article_row in tqdm(articles_df.iterrows()):
        sentences = _split_into_sentences(
            article_row["text"],
            article_row["title"],
        )
        sentence_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
        sentence_dataframes.append(
            pd.DataFrame(
                {
                    "sentence": sentences,
                    "article_id": [article_row["article_id"]] * len(sentences),
                    "sentence_id": sentence_ids,
                },
            ),
        )

    df = pd.concat(sentence_dataframes)

    return df


if __name__ == "__main__":
    main()
