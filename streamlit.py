import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, JsCode,GridOptionsBuilder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Music Recommendation App", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(suppress_st_warning=True)
def load_data():
    artists_names=pd.read_pickle('new_data/artist_names.pkl')
    song_meta=pd.read_pickle('new_data/song_meta.pkl')
    unique_artist_id=pd.read_pickle('new_data/unique_artist_id.pkl')
    unique_artists_names = pd.DataFrame(unique_artist_id).merge(artists_names, on='artist_id', how='inner')[[
        'artist_name','artist_id']]
    track_genre=pd.read_pickle('new_data/track_genre.pkl')
    artist_similarity=pd.read_pickle('new_data/artist_similarity.pkl')
    return artists_names,song_meta,unique_artist_id,unique_artists_names,track_genre,artist_similarity

def load_data2():
    artists_tag = pd.read_pickle('new_data/artist_tags.pkl')
    return artists_tag

def load_data3():
    artist_hotness = pd.read_pickle('new_data/artist_plays_hotness.pkl')
    artist_location = pd.read_pickle('new_data/artist_location.pkl')
    artist_genre = pd.read_pickle('new_data/artist_genre.pkl')
    track_play = pd.read_pickle('new_data/track_play_summary.pkl')
    song_tags_summary = pd.read_pickle('new_data/song_tags_summary.pkl')
    return artist_hotness,artist_location,artist_genre,track_play,song_tags_summary

artists_names, song_meta, unique_artist_id,unique_artists_names,track_genre,artist_similarity = load_data()
artists_tag=load_data2()
artist_hotness,artist_location,artist_genre,track_play,song_tags_summary=load_data3()

unique_artists_names=unique_artists_names.merge(pd.DataFrame(artist_location['artist_id']).drop_duplicates(),on='artist_id',how='inner')

title_spacer1, main_title, title_spacer2, sub_title, title_spacer3 = st.columns(
    (0.1, 2.5, 0.1, 1, 0.1)
)

main_title.title('Music Recommendation & Visulization 🎶🎧🎸')
sub_title.subheader("⭐ DVA Course ⭐ Project Team 164 ⭐")

input_box_spacer1, selectbox, input_box_spacer2 = st.columns((0.1, 3.2, 0.1))

with selectbox:
    selected_artist = st.selectbox(
        "Select one of your favorite artist",
        (
            unique_artists_names['artist_name'].tolist()
        ),
    )

selected_artist_id=unique_artists_names[unique_artists_names['artist_name']==selected_artist]['artist_id'].values[0]

# vis row 1
st.write("")
chart_row1_spacer1, chart_row1_p1, chart_row1_spacer2, chart_row1_p2, chart_row1_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

text_row1_spacer1, text_row1_p1, text_row1_spacer2, text_row1_p2, text_row1_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

chart_row2_spacer1, chart_row2_p1,chart_row2_spacer2, chart_row2_p2, chart_row2_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

text_row2_spacer1, text_row2_p1,text_row2_spacer2, text_row2_p2, text_row2_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

chart_row3_spacer1, chart_row3_p1,chart_row3_spacer2, chart_row3_p2, chart_row3_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

text_row3_spacer1, text_row3_p1,text_row3_spacer2, text_row3_p2, text_row3_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

chart_row4_spacer1, chart_row4_p1,chart_row4_spacer2, chart_row4_p2, chart_row4_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

text_row4_spacer1, text_row4_p1,text_row4_spacer2, text_row4_p2, text_row4_spacer2 = st.columns(
    (0.1, 1, 0.02, 1, 0.1)
)

with chart_row1_p1:
    st.subheader("Artist Background")
    df = song_meta[song_meta['artist_id'] == selected_artist_id]
    num_songs=df['track_id'].count()
    filtered_location = artist_location[artist_location.artist_id == selected_artist_id]
    filtered_hotness = artist_hotness[artist_hotness.artist_id == selected_artist_id]
    if  len(filtered_location) == 1:
        artist_data = {}
        artist_data['lat'] = filtered_location['lat'].iloc[0]
        artist_data['long'] = filtered_location['long'].iloc[0]
        artist_data['location_name'] = filtered_location['location'].iloc[0]
        artist_data['name'] = filtered_location['artist_name'].iloc[0]
        artist_data['artist_id'] = selected_artist_id
        artist_data['total_plays'] = filtered_hotness['total_plays']

        artist_map = px.scatter_geo(artist_data,
                                    lat='lat',
                                    lon='long',
                                    size='total_plays',
                                    hover_name='name',
                                    text='name',
                                    projection = 'natural earth',
                                    center = {'lat':filtered_location['lat'].iloc[0],'lon':filtered_location['long'].iloc[0]})
        st.plotly_chart(artist_map, use_container_width=True)

with text_row1_p1:
    text_artist_bg = filtered_location['artist_name'].iloc[0] + ' comes from ' + filtered_location['location'].iloc[0] + '. ' + filtered_location['artist_name'].iloc[0] + ' produced ' +  \
        str(num_songs) + ' songs and on average each song had been played by ' + str(round((artist_data['total_plays']/num_songs).iloc[0])) + ' times, with an overall hottness level of ' + str(filtered_hotness['play_rank'].iloc[0]) + ' in our database (1 being the hottest, 5 the least hottest).'
    st.markdown(text_artist_bg)

with chart_row1_p2:
    st.subheader("Years of Active (# of songs)")
    song_count = df.groupby("year")["track_id"].count().reset_index(name="count")
    year_tbl = pd.DataFrame(
        {'year': (
            range(min(song_count[song_count['year'] > 0]['year'], default=0), max(song_count['year']) + 1))}).append(
        {'year': 0}, ignore_index=True).drop_duplicates()
    song_cnt_update = year_tbl.merge(song_count, on='year', how='left')
    song_cnt_update.loc[song_cnt_update.year == 0, 'year'] = 'Unknown'
    song_cnt_update.loc[song_cnt_update.year == 'Unknown', 'color'] = 'Unknown'
    song_cnt_update.loc[song_cnt_update.year != 'Unknown', 'color'] = 'Known'
    song_cnt_update.loc[song_cnt_update.year == 'Unknown', 'count_bar'] = 1
    song_cnt_update.loc[song_cnt_update.year != 'Unknown', 'count_bar'] = song_cnt_update.loc[
        song_cnt_update.year != 'Unknown', 'count']
    fig = px.bar(song_cnt_update, x="year", y="count_bar", text='count', color='color',
                 color_discrete_map={'Unknown': 'grey',
                                     'known': 'green'},
                 hover_data={'count_bar': False,
                             'color': False}
                 )
    fig.update_layout(xaxis_title='year',
                      yaxis_title='number of songs released',
                      showlegend=False)
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)


with text_row1_p2:
    if (len(song_cnt_update) == 1) & (song_cnt_update.loc[0, 'year'] == 'Unknown'):
        text_1 = selected_artist + ' released ' + str(song_cnt_update.loc[
                                                         0, 'count']) + " songs according to our database. Unfortunately, we don't have any year information about these songs"
    else:
        song_cnt_update_filtered=song_cnt_update[song_cnt_update['year'] != 'Unknown']
        mostyear = song_cnt_update_filtered[song_cnt_update_filtered['count'] == max(song_cnt_update_filtered['count'])]['year'].iloc[0]
        text_1 = selected_artist + ' released ' + str(
            round(song_cnt_update_filtered.loc[:, 'count'].sum())) + " songs between year " + \
                 str(min(song_count[song_count['year'] > 0]['year'], default=0)) + " and 2010"  + \
                 '. The most active year of ' + selected_artist + ' is ' + str(mostyear) + ', during when ' + str(
            round(max(song_cnt_update_filtered['count']))) + ' songs had been produced.'
    st.markdown(text_1)

with chart_row2_p1:
    st.subheader("Genre of Artist's Songs")
    df = song_meta[song_meta['artist_id'] == selected_artist_id]
    song_tags_summary_sample = df.merge(track_genre, on='track_id', how='inner').groupby('major_genre')[
        'track_id'].count().reset_index()
    song_tags_summary_sample.columns = ['Genre', 'Number of Songs']
    fig = px.treemap(song_tags_summary_sample, path=[px.Constant("All Genres"), 'Genre'], values='Number of Songs')
    fig.update_layout(margin=dict(l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

with text_row2_p1:
    song_tags_summary_sample['percent']=song_tags_summary_sample['Number of Songs']/sum(song_tags_summary_sample['Number of Songs'])
    major_genre=song_tags_summary_sample[song_tags_summary_sample['Number of Songs']==max(song_tags_summary_sample['Number of Songs'])]['Genre'].iloc[0]
    major_genre_percent=song_tags_summary_sample[song_tags_summary_sample['Number of Songs']==max(song_tags_summary_sample['Number of Songs'])]['percent'].iloc[0]
    text_2 = 'Majority of (' + str(round(major_genre_percent*100)) + '% of all) '  + selected_artist + "'s songs are of genre "+ major_genre + '.'
    st.markdown(text_2)

with chart_row2_p2:
    st.subheader("Social Media tags of the artists")
    stopwords = set(STOPWORDS)
    selected_tags=artists_tag[artists_tag['artist_id']== selected_artist_id]['artist_tag'].tolist()
    text=" ".join(selected_tags)
    wc = WordCloud(background_color="white", colormap="hot",stopwords=stopwords,width=800, height=400)
    # generate word cloud
    wc.generate(text)
    plt.imshow(wc, interpolation='bilinear')
    # Display the generated image:
    plt.axis("off")
    st.pyplot()

with text_row2_p2:
    text_wordcloud=  selected_artist + ' is mostly described as ' + ', '.join(selected_tags[0:10]) + '.'
    st.markdown(text_wordcloud)

# vis row 2
with chart_row3_p1:
    st.subheader("Recommended Similar Artists")
    filtered_artist = \
    artist_similarity[artist_similarity['artist_id'] == selected_artist_id].merge(unique_artists_names,
                                                                                  left_on='similar_artist',
                                                                                  right_on='artist_id',
                                                                                  how='inner')[['similar_artist','artist_name', 'score']] \
        .sort_values(by='score', ascending=False)[0:20].merge(artist_genre,left_on = 'similar_artist', right_on='artist_id',how='left')
    df = filtered_artist.rename(columns={'score': 'Similarity Score',
                                         'artist_name': 'Similar Artist',
                                         'major_genre': 'Artist Genre'})
    fig = px.bar(df,x='Similar Artist',y='Similarity Score',color= 'Artist Genre')
    st.plotly_chart(fig, use_container_width=True)

with text_row3_p1:
    genre_of_recommended=df.groupby('Artist Genre')['artist_id'].count().reset_index().sort_values(by='artist_id', ascending=False)['Artist Genre'].iloc[0]
    text_similar_artist= 'Here are the most similar artists we recommend to you. Majorty of the recommended artists are ' + genre_of_recommended + ' artist.'
    st.markdown(text_similar_artist)

with chart_row3_p2:
    st.subheader("Genre of songs from recommended artists")

    track_genre_of_similar_aritist = filtered_artist[['similar_artist', 'artist_name']]. \
        merge(song_meta, left_on='similar_artist', right_on='artist_id', how='inner'). \
        merge(track_genre, on='track_id', how='inner'). \
        groupby(['artist_name', 'major_genre'])['track_id']. \
        count().reset_index()

    df2=track_genre_of_similar_aritist.rename(columns={'artist_name':'Similar Artist',
                                                      'track_id':'Number of Songs',
                                                      'major_genre':'Genre of Songs'})

    fig = px.bar(df2, x='Similar Artist', y='Number of Songs', color='Genre of Songs')
    st.plotly_chart(fig, use_container_width=True)

with text_row3_p2:
    text_track_of_similar_aritst='This chart shows the genre of songs from recommended artists.'
    st.markdown(text_track_of_similar_aritst)

with chart_row4_p1:
    st.subheader("Social Media tags of the similar artists")
    sub_selected_artist = st.selectbox(
        "Select one of the similar artist to generate an updated word cloud",
        (
            filtered_artist['artist_name'].tolist()
        ),
    )
    sub_selected_artist_id = \
    unique_artists_names[unique_artists_names['artist_name'] == sub_selected_artist]['artist_id'].values[0]

    stopwords = set(STOPWORDS)
    selected_tags = artists_tag[artists_tag['artist_id'] == sub_selected_artist_id]['artist_tag'].tolist()
    text = " ".join(selected_tags)
    wc = WordCloud(background_color="white", colormap="hot", stopwords=stopwords, width=800, height=400)
    # generate word cloud
    wc.generate(text)
    plt.imshow(wc, interpolation='bilinear')
    # Display the generated image:
    plt.axis("off")
    st.pyplot()

with text_row4_p1:
    text_wordcloud_2=  sub_selected_artist + ' is mostly described as ' + ', '.join(selected_tags[0:10]) + '.'
    st.markdown(text_wordcloud_2)

with chart_row4_p2:
    st.subheader("Recommended Songs")
    filtered_artist_top_10 = \
        artist_similarity[artist_similarity['artist_id'] == selected_artist_id].merge(unique_artists_names,
                                                                                      left_on='similar_artist',
                                                                                      right_on='artist_id',
                                                                                      how='inner')[
            ['similar_artist', 'artist_name', 'score']] \
            .sort_values(by='score', ascending=False)[0:10]

    song_summary_table = filtered_artist_top_10.merge(song_meta,
                                                      left_on='similar_artist',
                                                      right_on='artist_id',
                                                      how='inner')[
        ['artist_name', 'similar_artist', 'title', 'year', 'track_id', 'song_id']]. \
        merge(track_play, on='song_id', how='inner'). \
        merge(song_tags_summary, on='track_id', how='inner'). \
        merge(track_genre, on='track_id', how='inner')

    new_list = []
    for i in song_summary_table['hottness_level'].astype('int').tolist():
        new_list.append(i * '\U0001f525')
    song_summary_table['hot_emoji'] = new_list
    song_summary_table['search_term'] = song_summary_table['title'] + '+%2B+' + song_summary_table['artist_name']
    song_summary_table['search_term']='https://www.youtube.com/results?search_query='+song_summary_table['search_term'].str.replace(' ', '+', regex=False)

    final_tbl = song_summary_table[
        ['artist_name', 'title','search_term', 'year', 'count', 'play_freq', 'major_genre', 'hot_emoji','tags']]
    final_tbl=final_tbl.rename(columns={'artist_name':'Similar Artist',
                              'title':'Song Name',
                              'year' : 'Year',
                              'count' : 'Num of Plays',
                              'play_freq' : 'Plays Per User',
                              'major_genre' : 'Genre',
                              'hot_emoji':'Hotness',
                              'tags':'Tags',
                              'search_term':'Link'})
    gb=GridOptionsBuilder.from_dataframe(final_tbl)
    cell_renderer = JsCode("""
    function(params) {return `<a href=${params.value} target="_blank">🔗</a>`}
    """)
    gb.configure_column("Link", cellRenderer=cell_renderer,width=80)
    gb.configure_column("Year",width=80)
    gb.configure_column("Num of Plays",  width=130)
    gb.configure_column("Plays Per User", width=130)
    gb.configure_column("Genre", width=100)
    gb.configure_column("Tags", width=800)


    gridOptions=gb.build()

    AgGrid(final_tbl,gridOptions=gridOptions,allow_unsafe_jscode =True,height=450)


#selected_artist='2Pac'
#artists_names[artists_names['artist_name']==selected_artist]
#selected_artist_id='ARPXABO1187FB4D15F'
