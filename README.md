# Steam Project

## Introduction
This project simulated the role of an MLOps Engineer, combining the responsibilities of a Data Engineer and Data Scientist, to develop an API for the video game platform Steam. The goal was to create a Minimum Viable Product (MVP), implementing sentiment analysis on user comments and game recommendations based on the item-item model or the user-item model.

## Context
Steam is a digital distributor, digital rights management, multiplayer, and social networking service developed by Valve Corporation. Launched in September 2003, Steam is a popular platform for gamers of all ages, offering a wide variety of games and features.

## Transformations
We received a file named PI MLOps - STEAM-20231219T030929Z-001, containing three additional zip files:

- **australian_users_items.json:** Contains information about the games played by all users, including the accumulated playtime forever for each user. The "items" column was nested, and I had to update the code, using the final version in `1. ETL_users_items.ipynb`.

- **australian_user_reviews.json:** This file contained user reviews, but the "review" column was nested. I updated the code, using the final version in `2. ETL_user_reviews.ipynb`.

- **output_steam_games.json:** Contains data related to games, such as titles, developers, genres, release_date, and other columns. The "genres" column was nested, and the file was full of NAN data. I updated the code, using the final version in the file `3. ETL_steam_games.ipynb`.

For all these files, I found and removed null data, duplicates, removed unnecessary columns, and managed empty cells. You can find more details in every file.

## Exploratory Data Analysis (EDA)
After exploring and cleaning the files during the transformation process, I used `matplotlib/seaborn` in `4. EDA.ipynb` to analyze every file. This provided me with a general vision to build the sentiment analysis and the recommendation model.

## Sentiment Analysis
I applied `textblob` to build the sentiment analysis in the file `5. Sentiment_analysis.ipynb` that classifies sentiments on a scale of 0 for negative, 1 for neutral, and 2 for positive, creating a new column and a new CSV file.

## API Queries and Functions
In `6. API_queries.ipynb`, I designed all the datasets necessary to answer all the project functions. I created 3 files in total: 1 file for the functions `PlayTimeGenre` and `UserForGenre`, 1 file for `UsersRecommend`, `UsersWorstDeveloper`, and `sentiment_analysis`, and 1 file for the recommendation item-item system.

I tested all the functions in `7. API_functions.ipynb` before creating and deploying the API.

## API Development
I developed an API using the FastAPI framework with the following functions:

- **PlayTimeGenre(genre: str):** Should return the release year with the most played hours for the given genre.
- **UserForGenre(genre: str):** Should return the user with the highest accumulated playtime for the given genre and a list of playtime accumulation per year.
- **UsersRecommend(year: int):** Returns the top 3 games MOST recommended by users for the given year.
- **UsersWorstDeveloper(year: int):** Returns the top 3 developers with games LEAST recommended by users for the given year.
- **sentiment_analysis(developer_company: str):** According to the developer company, returns a dictionary with the company name as the key and a list with the total count of user review records categorized with sentiment analysis as the value.

For the recommendation system, I followed this:

- **game_recommendation(product_id):** Given the product ID, should return a list of 5 games recommended similar to the input.

## Deployment
For deploying the API, I used FastAPI, and the Render platform was chosen, which is a unified cloud service for creating and running applications and websites, allowing automatic deployment from GitHub.

The deployment process involved the following steps:

- **Service Configuration on Render.com:** Set up a new service on [render.com](https://render.com), connected to the current repository, and using [GitHub link here](https://github.com/ajalvarez75/PI-STEAM.git).

- **Final Deployment:** The service is now running at [https://pi-steam-render.onrender.com/docs](https://pi-steam-render.onrender.com/docs).

## Video
A brief explanation of this project, showcasing the API's functionality, is available in [this video](https://youtu.be/l54r0iMK0rE).