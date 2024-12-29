import streamlit as st
import pandas as pd
import pickle

# Load the trained model (replace 'model.pkl' with your actual model file)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title of the app
st.title("IPL Probability Predictor")

# Sidebar for input
st.sidebar.header("Input Match Details")

# Input fields for prediction
teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Rajasthan Royals', 'Sunrisers Hyderabad', 'Punjab Kings', 'Delhi Capitals', 'Lucknow Super Giants', 'Gujarat Titans']

stadiums = ['Wankhede Stadium', 'Chinnaswamy Stadium', 'Eden Gardens', 'Feroz Shah Kotla', 'Narendra Modi Stadium',
            'M. A. Chidambaram Stadium', 'Sawai Mansingh Stadium', 'Rajiv Gandhi International Stadium']

batting_team = st.sidebar.selectbox("Select Batting Team", teams)
bowling_team = st.sidebar.selectbox("Select Bowling Team", teams)
stadium = st.sidebar.selectbox("Select Stadium", stadiums)
current_score = st.sidebar.number_input("Current Score", min_value=0, step=1, value=100)
over = st.sidebar.slider("Over", min_value=0, max_value=20, value=10)
wickets = st.sidebar.slider("Wickets Lost", min_value=0, max_value=10, value=2)

# Button for prediction
if st.sidebar.button("Predict Probability"):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'stadium': [stadium],
        'current_score': [current_score],
        'over': [over],
        'wickets': [wickets]
    })

    # Make the prediction
    try:
        probabilities = model.predict_proba(input_data)
        win_probability = probabilities[0][1] * 100  # Assuming the second column corresponds to win probability
        lose_probability = probabilities[0][0] * 100

        # Display the result
        st.subheader("Match Outcome Probabilities")
        st.write(f"**Win Probability for {batting_team}: {win_probability:.2f}%**")
        st.write(f"**Lose Probability for {batting_team}: {lose_probability:.2f}%**")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Developed by [Dhivakar]")
