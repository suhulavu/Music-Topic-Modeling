# IMPORTS
import psycopg2
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pkg_resources
from symspellpy.symspellpy import SymSpell
import re
import nltk
from gensim.utils import simple_preprocess
import pickle
from configparser import ConfigParser
from tqdm import tqdm


def preprocess():
    """
    Preprocesses music data with the following methods:
        - Language Detection
        - Spell Check
        - Tokenization
        - Lemmatization
    """

    # DOWNLOADING DATA FROM DATABASE
    config = ConfigParser()
    config.read('../config/config.ini')
    hostname = config.get('DATABASE', 'hostname')
    username = config.get('DATABASE', 'username')
    password = config.get('DATABASE', 'password')
    database = config.get('DATABASE', 'database')

    connection = psycopg2.connect(
        host=hostname,
        user=username,
        password=password,
        dbname=database
    )

    command = "SELECT * FROM lyrics"
    df = pd.read_sql(command, connection)
    connection.close()


    # LANGUAGE DETECTION
    def checkEnglish(song):
        try:
            lang = detect(song)
            return lang == 'en'
        except LangDetectException:
            return False

    DetectorFactory.seed = 42
    df = df.loc[[checkEnglish(x) for x in tqdm(df['lyrics'], desc='Language Check')]]


    # SPELL CHECK
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename('symspellpy', "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    df['lyrics'] = [sym_spell.lookup_compound(lyrics, max_edit_distance=2)[0]._term for lyrics in tqdm(df['lyrics'], desc='Spell Check')]


    # TEXT PROCESSING
    # remove punctuation
    df['clean_lyrics'] = df['lyrics'].map(lambda x: re.sub('[,\.!?:]', '', x))

    stopwords = nltk.corpus.stopwords.words('english')

    # adding extra stopwords commonly found in song lyrics
    more_stopwords = ['ooh','yeah','hey','whoa','woah', 'ohh', 'was', 'mmm', 'oooh','yah','yeh','mmm', 'hmm','deh','doh','jah','wa', 'yuh',
                    'get', 'as', 'got', 'huh', 'one']
    stopwords.extend(more_stopwords)

    # tokenization
    corpus = [simple_preprocess(doc, deacc=True, min_len=3) for doc in df['clean_lyrics']]

    # removing stopwords and lemmatization
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    corpus = [[lem.lemmatize(token) for token in doc if token not in stopwords] for doc in corpus]


    # SAVING DATA
    df['clean_lyrics'] = corpus
    pickle.dump(df, open('../data/processed_data.pkl', 'wb'))



if __name__ == "__main__":
    preprocess()