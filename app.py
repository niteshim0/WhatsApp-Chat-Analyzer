import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import helper
import preprocessor
import seaborn as sns

st.title('WhatsApp Chat Analyzer')

uploaded_file = st.sidebar.file_uploader("Choose a file")


if uploaded_file is None:
    st.markdown('''Please export your WhatsApp chat (without media), whether it be a group chat or an individual/private chat, then click on "Browse Files" and upload it to this platform.''')
    st.markdown('''Afterward, kindly proceed to click on the "Analyse" button. This action will generate a variety of insights concerning your conversation.''')
    st.markdown(''' You will have the option to select the type of analysis, whether it is an overall analysis or one that specifically focuses on particular participants' analysis.''')
    st.markdown('Thank You!')
    st.markdown('Nitesh Kushwaha')

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.dataframe(df)

    #fetech unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button('Show Analysis'):

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
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color = '#FB9A99')
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
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)

        with col2:
            st.header('Most Active Month')
            most_active_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(most_active_month.index, most_active_month.values, color='#FDBF6F')
            ax.grid(False)
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)

        #Monthly
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

        #Most Common Words
        most_common_df = helper.most_common_words(selected_user, df)

        fig,ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color = '#1F78B4')
        ax.grid(False)
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        #emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title('Emoji Analysis')

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df.head())

        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), colors = ['#33A02C','#1F78B4','#B15928','#FB9A99','#FF7F00'], labels = emoji_df[0].head(), autopct = "%0.2f")
            st.pyplot(fig)

        #Daily Active user heatmap
        st.title('Weekly Activity Map')
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap, cmap='Paired')
        st.pyplot(fig)

        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
        st.text('by-Nitesh Kushwaha')
