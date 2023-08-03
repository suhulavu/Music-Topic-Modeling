# IMPORTS
import requests
from requests.exceptions import Timeout, ConnectionError, ConnectTimeout
from bs4 import BeautifulSoup
import psycopg2
import psycopg2.extras
from lyricsgenius import Genius
from collections import defaultdict
import pandas as pd
import json
import sys
from typing import List
import random
from configparser import ConfigParser

def getArtistData(artist:str, genius:Genius):
    """
    Compiles artist discography and metadata 

    Parameters
    ----------
    artist : str
        Artist's name
    genius : lyricsgenius.Genius
        Genius API connection object

    Returns
    -------
    dict 
        Artist's discography w/ metadata (song, lyrics, release dates)
    """

    print('Current Artist:', artist)

    # search for artist
    try:
        artist = genius.search_artist(artist, include_features=False)
    except (Timeout, ConnectTimeout, ConnectionError) as e:
        print('Failed:\n', e)
        return
    
    if artist:

        songs = artist.songs
        data = defaultdict(list)

        for song in songs:
            try:
                # lyrics
                lyrics = song.lyrics
                lyrics = lyrics[lyrics.index('\n') + 1:]
                if lyrics.endswith('Embed'):
                    idx = -6
                    while lyrics[idx].isnumeric():
                        idx -= 1
                    lyrics = lyrics[:idx + 1]
                lyrics = lyrics.encode('ascii', 'ignore').decode().lower().replace('\n', ' ')
                data['lyrics'].append(lyrics)

                # release date
                try:
                    year = song.to_dict()['release_date_components']['year']
                    data['year'].append(int(year))
                except:
                    data['year'].append(0)
                
                # song title
                title = song.title.encode('ascii', 'ignore').decode().lower()
                data['song'].append(title)
            except:
                continue

        return data
    
    return
    
def parseArtists(artists:List[str], genre:str):
    """
    Uploads artist data to PostgreSQL Server

    Parameters
    ----------
    artists: List[str]
        List of artists' names as strings
    genre: str
        Genre that the artists belong to
    """

    # reading config file
    config = ConfigParser()
    config.read('../config/config.ini')
    hostname = config.get('DATABASE', 'hostname')
    username = config.get('DATABASE', 'username')
    password = config.get('DATABASE', 'password')
    database = config.get('DATABASE', 'database')
    token = config.get('GENIUS API', 'client_access_token')

    # genius API connection
    genius = Genius(token, skip_non_songs=True, remove_section_headers=True, 
                    excluded_terms=['(Remix)', '(Live)', '(acoustic version)', '(piano version)'], verbose=False, sleep_time=1, retries=5)
    
    # postgresql server connection
    cnxn = psycopg2.connect(
        host=hostname,
        user=username,
        password=password,
        dbname=database,
    )
    cursor = cnxn.cursor()

    for artist in artists:
        # ensure artist name is formatted properly
        if '[' in artist:
            continue
        artist = artist[:artist.index('(') - 1] if '(' in artist else artist

        # check if artist data already exists in server
        check_command = "SELECT artist FROM lyrics WHERE artist = '{}'".format(artist.replace("'", "''"))
        cursor.execute(check_command)
        res = cursor.fetchone()
        if res:
            continue

        # parse artist data
        data = getArtistData(artist, genius)
        if data:
            data['artist'] = [artist for _ in range(len(data['lyrics']))]
            data['genre'] = [genre for _ in range(len(data['lyrics']))]
            df = pd.DataFrame(data)

            # insert data into table
            cols = ",".join(df.columns)
            placeholders = ",".join(['%s' for _ in range(len(df.columns))])
            insert_command = "INSERT INTO lyrics ({}) VALUES ({})".format(cols, placeholders)
            psycopg2.extras.execute_batch(cursor, insert_command, df.to_numpy())
            cnxn.commit()

    # close connection to server
    cursor.close()
    cnxn.close()



def scrapePop():
    """
    Scrapes all pop artists' discographies from https://today.yougov.com/ratings/entertainment/fame/pop-artists/all 
    """

    urls = ["https://today.yougov.com/_pubapis/v5/us/search/entity/?group=fba8bfe6-adef-11e9-8bb2-373b0b3b3eb4&sort_by=fame&limit=20&offset={}".format(i)
                  for i in range(0, 241, 20)]
    
    # extract artists from webpage
    for url in urls:
        page = requests.get(url)
        data = json.loads(page.text)['data']
        artists = [data[i]['url'].replace('_', ' ') for i in range(len(data))]
        page.close()
        parseArtists(artists=artists, genre='pop')


def scrapeOther(genre):
    """
    Scrapes artist discographies from the genres of hip hop, country, R&B, and rock from Wikipedia
    """

    urls = {
        'hip hop': "https://en.wikipedia.org/wiki/List_of_hip_hop_musicians",
        'country': "https://en.wikipedia.org/wiki/List_of_country_music_performers",
        'R&B': "https://en.wikipedia.org/wiki/List_of_R%26B_musicians",
        'rock': "https://en.wikipedia.org/wiki/List_of_hard_rock_musicians_(A%E2%80%93M)",
    }

    # scrape artists from webpage
    page = requests.get(urls[genre])
    soup = BeautifulSoup(page.content, 'html.parser')
    data = soup.select('.div-col ul a')
    artists = [x.text for x in data]
    random.shuffle(artists)
    page.close()
    parseArtists(artists=artists, genre=genre)


def scrapeMusic(genre):
    if genre not in ['pop', 'rock', 'hip hop', 'country', 'R&B']:
        raise RuntimeError('Genre not supported')
    
    scrapePop() if genre == 'pop' else scrapeOther(genre)

if __name__ == '__main__':
    scrapeMusic(sys.argv[1])