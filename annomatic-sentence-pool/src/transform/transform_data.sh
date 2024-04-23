#!/bin/bash

python transform/filter_articles.py
python transform/create_sentences.py
python transform/filter_sentences.py
python transform/create_final_dataset.py
