import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")


def is_noun(tag):
    return tag in ["NN", "NNS", "NNP", "NNPS"]


def is_verb(tag):
    return tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


def is_adverb(tag):
    return tag in ["RB", "RBR", "RBS"]


def is_adjective(tag):
    return tag in ["JJ", "JJR", "JJS"]


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None


def clean_token_text(token):
    """Clean the token text by lemmatizing it and striping whitespaces."""
    lemmatizer = WordNetLemmatizer()
    token = token.strip(" ")
    tag = penn_to_wn(nltk.pos_tag(tokens=[token])[0][1])
    if tag == None:
        return token

    return lemmatizer.lemmatize(token, tag)


def norm_(ar):
    """Normalize the (attention) array."""
    normalized = (ar - ar.min()) / (ar.max() - ar.min())
    return normalized / normalized.sum()


def remove_stopwords(sentence):
    stop_words = set(
        stopwords.words("english")
    )  # Change 'english' to your language of choice
    word_tokens = word_tokenize(sentence)

    filtered_sentence = [
        word for word in word_tokens if word.lower() not in stop_words
    ]

    # Join the words back into a sentence
    return " ".join(filtered_sentence)
