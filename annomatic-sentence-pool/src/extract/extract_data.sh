#!/bin/bash

python extract/scrape_allsides.py
python extract/merge_outlet_info.py
python extract/scrape_articles.py
