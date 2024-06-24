# MVP (Minimum Viable Product) for a news recommender systems
![Author](https://img.shields.io/badge/Author-Ahmed%20Ait%20Ouazzou-brightgreen)
[![GitHub](https://img.shields.io/badge/GitHub-Follow%20Me-lightgrey)](https://github.com/ahmedaao)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-informational)](https://www.linkedin.com/in/ahmed-ait-ouazzou/)

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Test Locally](#test-locally)
3. [Data Source](#data-source)
4. [Technologies](#technologies)
5. [Models](#models)
6. [Architecture](#architecture)
7. [Methods](#methods)
    - [Method 1: BeautifulSoup and Requests](#method-1-beautifulsoup-and-requests)
    - [Method 2: Scrapy](#method-2-scrapy)
    - [Method 3: Selenium](#method-3-selenium)

## Introduction

Numerous press articles are available on the web covering topics of all kinds (politics, sports, health, etc.). This directory provides a minimal application (MVP) that recommends articles to users based on their interests. To do this, we use a deep learning model that has been trained on user data from the Brazilian site 'Globo.com'.

## Installation

```sh
git clone git@bitbucket.org:ahmedaao/news-recommender-systems.git
cd news-recommender-systems
python3 -m venv -venv
pip install -r requirements.txt
pip install . # Install modules from package src/
```
## Test Locally

You can test the application locally in three ways:
    1. Directly from fastAPI and streamlit

    Go to the root of the repo, then:
    ```sh
    cd app/backend
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
    ```
    
    Now, open a new terminal and enter:
    ```sh
    cd app/frontend
    streamlit run streamlit_app.py
    ```

    2. Through docker
    3. Through docker-compose

## Data Source

To accomplish this project, we use dataset at this [link](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom#clicks_sample.csv)

## Technologies

- Python
- surprise (make recommender systems)
- streamlit (frontend)
- fastApi (middleware)
- docker
- docker-compose
- AWS EC2
- AWS Lambda

## Models

In this project we will use the following models: 

- Content-based filtering
- Collaborative filtering
    1. Model clustering based (algorithm KNN)
    2. Model matrix factorization based (algorithm SVD)

## Architecture


## Methods

### Method 1: BeautifulSoup and Requests

This method utilizes the popular BeautifulSoup library for parsing HTML and the Requests library for making HTTP requests. It is a simple and effective approach for extracting data from static web pages.
