from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os

app = FastAPI()
# uvicorn main:app --reload

# Configure file paths using environment variables
DATA_DIR = os.getenv('DATA_DIR', 'datasets')
FUNCTION_1_2_FILE = os.path.join(DATA_DIR, 'function_1_2.csv')
FUNCTION_3_4_5_FILE = os.path.join(DATA_DIR, 'function_3_4_5.csv')
FUNCTION_ITEM_ITEM_FILE = os.path.join(DATA_DIR, 'function_item_item.csv')

# Load files
df_function_1_2 = pd.read_csv(FUNCTION_1_2_FILE)
df_function_3_4_5 = pd.read_csv(FUNCTION_3_4_5_FILE)
df_function_item_item = pd.read_csv(FUNCTION_ITEM_ITEM_FILE)

# First function.
@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str):
    genre = genre.lower()

    # Convert playtime to hours
    df_function_1_2['playtime_forever'] = df_function_1_2['playtime_forever'] / 60

    # Filter DataFrame based on genre
    df_filtered = df_function_1_2[df_function_1_2['genres'] == genre]

    # Group by release year and sum playtime
    grouped = df_filtered.groupby(['year'])['playtime_forever'].sum().reset_index()

    # Find the release year with the maximum playtime
    max_played_year = grouped.loc[grouped['playtime_forever'].idxmax()]['year']

    result = {"Year of release with the most playtime for Genre " + genre: int(max_played_year)}

    return result

# Second function.
@app.get("/UserForGenre/{genre}")
def UserForGenre(genre: str):
    df_filter = df_function_1_2[df_function_1_2['genres'].str.lower() == genre.lower()]

    if df_filter.empty:
        return {"Message": f"No data found for the genre {genre}"}

    # Group by user and release year
    grouped = df_filter.groupby(['user_id', 'year'])['playtime_forever'].sum().reset_index()
    grouped['playtime_forever'] = grouped['playtime_forever'] / 60

    # User + hours played per genre
    max_user = grouped.groupby('user_id')['playtime_forever'].sum().idxmax()
    max_user_hours = grouped[grouped['user_id'] == max_user]

    # Get the list of accumulated hours played per year
    hours_per_year = max_user_hours[['year', 'playtime_forever']]
    hours_list = [{"Year": int(year), "Hours": int(hours)} for year, hours in zip(hours_per_year['year'], hours_per_year['playtime_forever'])]

    result = {"User with the most hours played for Genre " + genre: max_user, "Hours played": hours_list}
    return result

# Third function.
@app.get("/UsersRecommend/{year}")
def UsersRecommend(year: int):
    # Filter the DataFrame based on the given year
    df_filter = df_function_3_4_5[df_function_3_4_5['year'] == year]

    # Check if the filtered DataFrame is empty
    if df_filter.empty:
        return {"Message": f"No recommended games found for the year {year}"}

    # Filter games with a recommendation score of 2 and group by title
    recommended_games = df_filter[df_filter['reviews_recommend'] == 2].groupby('title').size().reset_index()

    # Rename the count column and sort the DataFrame by count in descending order
    top_games = recommended_games.rename(columns={0: 'count'}).sort_values(by='count', ascending=False).head(3)

    # Create a result list with game positions and titles
    result = [{"Position " + str(i + 1): game} for i, game in enumerate(top_games['title'])]

    return result

# Fourth function.
@app.get("/UsersWorstDeveloper/{year}")
def UsersWorstDeveloper(year: int):
    # Filter by year and games with reviews_recommend equal to 1
    df_filtered = df_function_3_4_5[(df_function_3_4_5['year'] == year) & (df_function_3_4_5['reviews_recommend'] == 1)]

    if df_filtered.empty:
        return [{"Message": f"No data found for the year {year}"}]

    # Count the frequency of games with reviews_recommend equal to 1 per developer
    developer_counts = df_filtered['developer'].value_counts()

    # Get the top 3 developers with the most games with reviews_recommend equal to 1
    top3_worst_developers = developer_counts.nlargest(3).index.tolist()

    # Create the output in the specified format
    result = [{"Puesto 1": top3_worst_developers[0]}, {"Puesto 2": top3_worst_developers[1]}, {"Puesto 3": top3_worst_developers[2]}]
    return result

# Fifth function.
@app.get("/sentiment_analysis/{year}")
def sentiment_analysis(developer_company: str):

    # Convert developer_company to lowercase
    developer_company = developer_company.lower()

    df_filtered = df_function_3_4_5[df_function_3_4_5['developer'] == developer_company]

    if df_filtered.empty:
        return {developer_company: ['Negative = 0, Neutral = 0, Positive = 0']}

    # Count the number of review records for each sentiment analysis
    sentiment_counts = df_filtered['sentiment_analysis'].value_counts()

    # Create the return dictionary in the specified format
    result = {
        developer_company: [
            f"Negative = {sentiment_counts.get(0, 0)}, Neutral = {sentiment_counts.get(1, 0)}, Positive = {sentiment_counts.get(2, 0)}"
        ]
    }

    return result

# Define the item-item game recommendation function.
@app.get("/game_recommendation/{product_id}")
def game_recommendation(product_id: str):
    # Convert the product_id to int
    try:
        product_id_int = int(product_id)
    except ValueError:
        return f'Invalid game ID: {product_id}'

    # Filter the DataFrame to get the input game vector.
    input_game = df_function_item_item[df_function_item_item['item_id'] == product_id_int].drop(['item_id', 'title', 'genres'], axis=1)

    if input_game.empty:
        return f'No game found with ID {product_id}'

    # Filter games that the user has already played before calculating cosine similarity.
    df_function_item_item_filtered = df_function_item_item[~df_function_item_item['item_id'].isin([product_id_int])]

    # Calculate the cosine similarity between the input game and all other games.
    similarities = cosine_similarity(input_game, df_function_item_item_filtered.drop(['item_id', 'title', 'genres'], axis=1))

    # Get the indices of the most similar games.
    similar_indices = similarities.argsort()[0][-10:][::-1]

    # Get the list of recommended games excluding the ones already played.
    recommended_games = []
    for idx in similar_indices:
        recommended_game = df_function_item_item_filtered.iloc[idx][['item_id', 'title', 'genres']].to_dict()
        if recommended_game not in recommended_games:
            recommended_games.append(recommended_game)
        if len(recommended_games) == 5:
            break

    return recommended_games