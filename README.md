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
4. Choose the configuration file: under the ` configuration folder `, it is possible to choose from which data run the system (Planeat.eco or Food.com). In order to change the data it is necessary to rename the selected file as ` config.json `. From the ` config.json ` file the system will gather all the data it needs without any further configuration. 

## Execution 
* In order to run the streamlit web application:
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
