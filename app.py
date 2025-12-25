import streamlit as st
import pandas as pd
import numpy as np
import joblib

## Load Model and Artifacts
artifacts = joblib.load("xgb_model.pkl")

xgb_model = artifacts['model']
sc = artifacts['scaler']
mlb = artifacts['mlb']
le_color = artifacts['le_color']
le_language = artifacts['le_language']
le_country = artifacts['le_country']
le_content_rating = artifacts['le_content_rating']

num_col=['num_critic_for_reviews',
 'duration',
 'director_facebook_likes',
 'gross',
 'num_voted_users',
 'cast_total_facebook_likes',
 'num_user_for_reviews',
 'budget',
 'title_year',
 'movie_facebook_likes']
## Streamlit App
st.set_page_config(page_title="IMDb Movie Score Predictor", layout="centered")
st.title("IMDb Movie Rating Predictor")
st.write("Predict IMDb score based on movie features")

## Input Features
color = st.selectbox("Color", le_color.classes_)
num_critic_for_reviews = st.number_input("Number of Critic Reviews", min_value=0)
duration = st.number_input("Duration (minutes)", min_value=0)
director_facebook_likes = st.number_input("Director Facebook Likes", min_value=0)
gross = st.number_input("Gross Revenue", min_value=0.0)
genres = st.multiselect("Genres", mlb.classes_)
num_voted_users = st.number_input("Number of Voted Users", min_value=0)
cast_total_facebook_likes = st.number_input("Cast Facebook Likes", min_value=0)
num_user_for_reviews = st.number_input("Number of User Reviews", min_value=0)
language = st.selectbox("Language", le_language.classes_)
country = st.selectbox("Country", le_country.classes_)
content_rating = st.selectbox("Content Rating", le_content_rating.classes_)
budget = st.number_input("Budget", min_value=0.0)
title_year = st.number_input("Title Year", min_value=1916, max_value=2016)
movie_facebook_likes = st.number_input("Movie Facebook Likes", min_value=0)


# Prediction

if st.button("Predict IMDb Rating"):
    
    # Encode categorical columns
    color_enc = le_color.transform([color])[0]
    language_enc = le_language.transform([language])[0]
    country_enc = le_country.transform([country])[0]
    content_rating_enc = le_content_rating.transform([content_rating])[0]

    # Encode genres (MultiLabelBinarizer)
    genres_encoded = mlb.transform([genres])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

    # Create base input dataframe
    input_df = pd.DataFrame([{
        'color': color_enc,
        'num_critic_for_reviews': num_critic_for_reviews,
        'duration': duration,
        'director_facebook_likes': director_facebook_likes,
        'gross': gross,
        'num_voted_users': num_voted_users,
        'cast_total_facebook_likes': cast_total_facebook_likes,
        'num_user_for_reviews': num_user_for_reviews,
        'language': language_enc,
        'country': country_enc,
        'content_rating': content_rating_enc,
        'budget': budget,
        'title_year': title_year,
        'movie_facebook_likes': movie_facebook_likes
    }])

    # Combine with genre features
    final_input = pd.concat([input_df, genres_df], axis=1)

    # Scale numeric features
    final_input[num_col] = sc.transform(final_input[num_col])

    # Predict
    prediction = xgb_model.predict(final_input)[0]

    st.success(f"⭐️ Predicted IMDb Rating: **{prediction:.2f}**")


# Footer

st.markdown("---")
st.caption("Built with XGBoost & Streamlit ")
