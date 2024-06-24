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

### Directly from fastAPI and streamlit

Go to the root of the repo, then:
```sh
cd app/backend
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

Now, open a new terminal, go back to the root of the repo and enter:
```sh
cd app/frontend
streamlit run streamlit_app.py
```

### Through docker 

Prerequisite: 

- Into requirements.txt file, replace 'ssh' with 'https' on the line starting with -e.
- Into streamlit.py, ip adress for backend has to be 0.0.0.0

Go to the root of the repo, first of all, create the docker image linked to fastapi

```sh
sudo docker build -t fastapi_app -f app/backend/Dockerfile .
```
Now, you can launch the docker container linked to fastapi with:
```sh
sudo docker run -d -p 8000:8000 --name fastapi_app_container fastapi_app:latest
```

Go back to the root of the repo. Now, create the docker image linked to streamlit
```sh
sudo docker build -t streamlit_app -f app/frontend/Dockerfile .
```
Now, you can launch the docker container linked to streamlit with:
```sh
sudo docker run -d -p 8501:8501 --name streamlit_app_container streamlit_app:latest
```

### Through docker-compose

Prerequisite: 

- Into streamlit.py, ip adress for backend has to be backend

Go to the root of the repo, then:
```sh
sudo docker-compose up -d
```

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
