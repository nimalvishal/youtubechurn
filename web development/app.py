from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
file_path = os.path.join(current_directory, 'E:/socket/web development/pythonfiles/dataset1.csv')

# Read the CSV file into a DataFrame
user_data = pd.read_csv(file_path)

data = {
    'Category': ['Category = 0', 'Category = 1'],
    'Unsubscribed': [50, 50]
}

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == 'channel123@gmail.com' and password == '123456':
        return redirect(url_for('onscreen'))
    else:
        return 'INVALID!'

@app.route('/onscreen')
def onscreen():  
    # Calculate the total number of YouTube videos and total stream time
    num_of_sub = user_data['Subscriber_id'].count()
    # Calculate genre frequencies
    genre_frequencies = user_data['Genre'].value_counts()
    # Prepare genre count information for rendering
    genre_info = [{'genre': genre, 'count': count} for genre, count in zip(genre_frequencies.index, genre_frequencies.values)]
    # Sort genre_info by count in descending order
    genre_info = sorted(genre_info, key=lambda x: x['count'], reverse=True)
    # Get the top 2 most frequently watched genres
    top2_genres = genre_info[:2]
    active_members = len(user_data[user_data['IsActiveMember'] == 1])
    # Calculate the count of inactive members (IsActiveMember = 0)
    inactive_members = len(user_data[user_data['IsActiveMember'] == 0])
    # Create a dictionary to store the counts
    member_counts = {
        'Active Members': active_members,
        'Inactive Members': inactive_members
    }
    # Calculate the total likes and total dislikes
    total_likes = user_data['Like_Dislike'].sum()
    total_dislikes = len(user_data) - total_likes
    top3_streamedtime = user_data.nlargest(3, 'Streamedtime')[['Subscriber_id', 'Streamedtime']]
    # Pass these random numbers and the combined total to the template
    return render_template('onscreen.html', numn_of_videos=num_of_sub, top3_streamedtime=top3_streamedtime, genree=top2_genres, member_counts=member_counts, random_likes=total_likes, random_dislikes=total_dislikes)

@app.route('/predict')
def predict():
    user_data= pd.DataFrame(data)
    # Create the pie chart using Plotly Express
    target_instance = user_data["Unsubscribed"].value_counts().to_frame()
    target_instance = target_instance.reset_index()
    target_instance = target_instance.rename(columns={'index': 'Category'})
    fig = px.pie(target_instance, values='Unsubscribed', names='Category', color_discrete_sequence=["green", "red"],
                title='Distribution of Churn')
    # Convert the Plotly figure to HTML
    pie_chart_html = pio.to_html(fig, full_html=False)   
    # Pass the HTML of the pie chart to the predict.html template
    return render_template('predict.html', piechart = pie_chart_html)

if __name__ == '__main__':
    app.run(debug=True)
