# Recommender System for reducing water footprint 🌎

### An health and planet aware recommender system for reducing water footprint of users' diet

This repository contains a recommender system that takes into account water footprint to suggest recipes to users. 
It provides a command line utility to explore the recommender system from terminal, and also, provides a Streamlit application to explore and configure all the possibility inserted into the recommender system. 
Before running the system it is possible to configure it via the application, you can choose around a content based or a collaborative filtering algorithm, the number of recommendations and the choice to activate or deactivate the water footprint filter. 

Data can be downloaded here: https://www.kaggle.com/turconiandrea/water-footprint-recommender-system-data

## Setup
1. Clone the application in a local folder

2. Download the dataset from the following links 
   * embbedding (ready to use): [embedding-data folder](https://www.kaggle.com/turconiandrea/water-footprint-recommender-system-data)

3. Paste the downloaded folder into ` data/ ` folder

## Execution 
* In order to run the streamlit application:
```bash
streamlit run app.py
```
* In order to run the application from command line (it is necesary the user id)
```bash
python main.py --user-id 52543 --algo cb
```
Arguments:
* user-id: the id of the user.
* algo: the algorithm type (cf or cb)
* no-filter-wf: to run the recommendation disabling the water footprint filter
