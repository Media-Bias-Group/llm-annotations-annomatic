import logging
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from src.utils_ import to_parquet

INPUT_PATH = "data/raw"
OUTPUT_PATH = "data/extract/tmp"

def _get_news_website(url):
    """
    Given a news article URL, returns the website link of the news source.

    Args:
        url (str): The URL of the news article.

    Returns:
        str: The website link of the news source, or None if not found.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the website link
    link_element = soup.find(
        "a",
        attrs={"class": "black-link", "target": "_blank"},
    )

    if link_element:
        website_link = link_element.get("href")
        return website_link
    else:
        return None

@to_parquet(f"{OUTPUT_PATH}/allsides_snapshot.parquet")
def main():
    """
    Scrapes media bias ratings from a HTML file and returns a pandas DataFrame.

    Args:
        path_to_file (str): The path to the HTML file to scrape.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped data with columns:
            - news_source (str): The name of the news source.
            - news_link (str): The link to the news source's website.
            - bias_rating (str): The media bias rating of the news source.
            - community_feedback (str): The community feedback.
    """
    logging.info("Parsing html file...")
    with open(f"{INPUT_PATH}/all_sides_snapshot_15_11_2023.html", "r", encoding="utf-8") as file:
        contents = file.read()
    soup = BeautifulSoup(contents, "html.parser")

    # Find the table with the media bias ratings
    table = soup.find("table", {"class": "views-table"})

    data = []

    logging.info("Scraping data...")
    # Iterate over each row of the table
    for row in tqdm(table.find_all("tr")[1:]):  # Skip the header row
        cols = row.find_all("td")
        if cols:
            news_source = cols[0].text.strip()
            bias_rating = (
                cols[1].find("img").attrs.get("alt", "").split(": ")[-1]
            )
            link = _get_news_website(cols[0].find("a").attrs.get("href", ""))
            community_feedback = re.search(r"\d+/\d+", cols[3].text)
            if community_feedback:
                community_feedback = community_feedback.group()
            else:
                community_feedback = None
            data.append([news_source, link, bias_rating, community_feedback])

    return pd.DataFrame(
        data,
        columns=[
            "news_source",
            "news_link",
            "bias_rating",
            "community_feedback",
        ],
    )


if __name__ == "__main__":
    main()
