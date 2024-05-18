import streamlit as st
import re
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

plt.style.use('dark_background')
import helper
import preprocessor
import seaborn as sns
import pandas as pd
import numpy as np
import nltk


# Function to perform sentiment analysis
def perform_sentiment_analysis(data):
    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()
    data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]]
    data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]]
    data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]]

    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0

    data['value'] = data.apply(lambda row: sentiment(row), axis=1)
    return data


# App title
st.title('WhatsApp Chat Analyzer')

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is None:
    st.markdown(
        '''Please export your WhatsApp chat (without media), whether it be a group chat or an individual/private chat, then click on "Browse Files" and upload it to this platform.''')
    st.markdown(
        '''Afterward, kindly proceed to click on the "Show Analysis" button. This action will generate a variety of insights concerning your conversation.''')
    st.markdown('Thank You!')
    st.markdown('Nitesh Kushwaha')

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button('Show Analysis'):
        # Fetching statistics
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title('Top Statistics')
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header('Total Messages')
            st.title(num_messages)
        with col2:
            st.header('Total Words')
            st.title(words)
        with col3:
            st.header('Media Shared')
            st.title(num_media_messages)
        with col4:
            st.header('Links Shared')
            st.title(num_links)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='#FB9A99')
        ax.grid(False)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header('Most Active Day')
            most_active_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(most_active_day.index, most_active_day.values, color='#FF7F00')
            ax.grid(False)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header('Most Active Month')
            most_active_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(most_active_month.index, most_active_month.values, color='#FDBF6F')
            ax.grid(False)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'])
        ax.grid(False)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Find the person who sent the most messages
        if selected_user == 'Overall':
            st.title('Most Active Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            ax.grid(False)

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='#6A3D9A')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Most Common Words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color='#1F78B4')
        ax.grid(False)
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        # Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title('Emoji Analysis')

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df.head())

        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), colors=['#33A02C', '#1F78B4', '#B15928', '#FB9A99', '#FF7F00'],
                   labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        # Daily Active user heatmap
        st.title('Weekly Activity Map')
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap, cmap='Paired')
        st.pyplot(fig)

        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Sentiment Analysis
        st.markdown("<h1 style='text-align: center; color: grey;'>Let's put some emotions in it</h1>",
                    unsafe_allow_html=True)


        # Function to perform sentiment analysis
        def perform_sentiment_analysis(data):
            nltk.download('vader_lexicon')
            sentiments = SentimentIntensityAnalyzer()
            data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]]
            data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]]
            data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]]

            def sentiment(d):
                if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
                    return 1
                if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
                    return -1
                if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
                    return 0

            data['value'] = data.apply(lambda row: sentiment(row), axis=1)
            return data

        # Perform sentiment analysis
        df_sentiment = perform_sentiment_analysis(df)

        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",
                        unsafe_allow_html=True)
            busy_month = helper.month_activity_mapsenti(selected_user, df_sentiment, 1)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)
            busy_month = helper.month_activity_mapsenti(selected_user, df_sentiment, 0)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",
                        unsafe_allow_html=True)
            busy_month = helper.month_activity_mapsenti(selected_user, df_sentiment, -1)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",
                        unsafe_allow_html=True)
            busy_day = helper.week_activity_mapsenti(selected_user, df_sentiment, 1)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",
                        unsafe_allow_html=True)
            busy_day = helper.week_activity_mapsenti(selected_user, df_sentiment, 0)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",
                        unsafe_allow_html=True)
            busy_day = helper.week_activity_mapsenti(selected_user, df_sentiment, -1)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",
                            unsafe_allow_html=True)
                user_heatmap = helper.activity_heatmapsenti(selected_user, df_sentiment, 1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",
                            unsafe_allow_html=True)
                user_heatmap = helper.activity_heatmapsenti(selected_user, df_sentiment, 0)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",
                            unsafe_allow_html=True)
                user_heatmap = helper.activity_heatmapsenti(selected_user, df_sentiment, -1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",
                        unsafe_allow_html=True)
            daily_timeline = helper.daily_timelinesenti(selected_user, df_sentiment, 1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)
            daily_timeline = helper.daily_timelinesenti(selected_user, df_sentiment, 0)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",
                        unsafe_allow_html=True)
            daily_timeline = helper.daily_timelinesenti(selected_user, df_sentiment, -1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",
                        unsafe_allow_html=True)
            timeline = helper.monthly_timelinesenti(selected_user, df_sentiment, 1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",
                        unsafe_allow_html=True)
            timeline = helper.monthly_timelinesenti(selected_user, df_sentiment, 0)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",
                        unsafe_allow_html=True)
            timeline = helper.monthly_timelinesenti(selected_user, df_sentiment, -1)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        # Percentage contributed
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                x = helper.percentagesenti(df, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                y = helper.percentagesenti(df, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                z = helper.percentagesenti(df, -1)

                # Displaying
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",
                            unsafe_allow_html=True)

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # WORDCLOUD......
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",
                            unsafe_allow_html=True)

                # Creating wordcloud of positive words
                df_wc = helper.create_wordcloudsenti(selected_user, df, 1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col2:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",
                            unsafe_allow_html=True)

                # Creating wordcloud of neutral words
                df_wc = helper.create_wordcloudsenti(selected_user, df, 0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col3:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",
                            unsafe_allow_html=True)

                # Creating wordcloud of negative words
                df_wc = helper.create_wordcloudsenti(selected_user, df, -1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')

        # Most common positive words
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # Data frame of most common positive words.
                most_common_df = helper.most_common_wordssenti(selected_user, df, 1)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col2:
            try:
                # Data frame of most common neutral words.
                most_common_df = helper.most_common_wordssenti(selected_user, df, 0)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col3:
            try:
                # Data frame of most common negative words.
                most_common_df = helper.most_common_wordssenti(selected_user, df, -1)

                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')




