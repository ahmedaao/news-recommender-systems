# MVP (Minimum Viable Product) for a news recommender systems
![Author](https://img.shields.io/badge/Author-Ahmed%20Ait%20Ouazzou-brightgreen)
[![GitHub](https://img.shields.io/badge/GitHub-Follow%20Me-lightgrey)](https://github.com/ahmedaao)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-informational)](https://www.linkedin.com/in/ahmed-ait-ouazzou/)

## Table of Contents

1. [Introduction](#introduction)
2. [Data Source](#data-source)
3. [Architecture](#architecture)
4. [Methods](#methods)
    - [Method 1: BeautifulSoup and Requests](#method-1-beautifulsoup-and-requests)
    - [Method 2: Scrapy](#method-2-scrapy)
    - [Method 3: Selenium](#method-3-selenium)
    - [Method 4: Requests and lxml](#method-4-requests-and-lxml)
    - [Method 5: LangChain](#method-5-langchain)

## Introduction


Numerous press articles are available on the web covering topics of all kinds (politics, sports, health, etc.). This directory provides a minimal application (MVP) that recommends articles to users based on their interests. To do this, we use a deep learning model that has been trained on user data from the Brazilian site 'Globo.com'.

## Data Source

To accomplish this project, we use dataset at this [link](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom#clicks_sample.csv)

## Architecture


## Methods

### Method 1: BeautifulSoup and Requests

This method utilizes the popular BeautifulSoup library for parsing HTML and the Requests library for making HTTP requests. It is a simple and effective approach for extracting data from static web pages.

### Method 2: Scrapy

Scrapy is a robust and extensible web scraping framework. It provides a complete solution for crawling websites and extracting structured data. This method is suitable for handling complex scraping tasks and building scalable spiders.

### Method 3: Selenium

Selenium is primarily used for browser automation, but it can also be employed for web scraping dynamic content. This method allows interaction with JavaScript-driven websites and provides a more dynamic approach to data extraction.

### Method 4: Requests and lxml

Combining the Requests library with lxml allows for efficient parsing of HTML and XML documents. This method is particularly useful for projects requiring speed and simplicity.

### Method 5: LangChain

The fifth method introduces LangChain, a powerful language model-based approach to web scraping. Leveraging advanced natural language processing capabilities, LangChain enhances the extraction of information from diverse sources.