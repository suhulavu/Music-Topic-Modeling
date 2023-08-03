# IMPORTS
import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.models import LdaMulticore, phrases
from gensim.models import CoherenceModel
from itertools import product
from collections import defaultdict
from tqdm import tqdm
import pickle


def calcCoherence(num_topics, alpha, beta, id2word, doc_freq, corpus):
    """
    Calculates coherence score for an LDA model

    Parameters
    -----------
    num_topics : int
        the number of topics for the LDA model
    alpha : float
        the alpha parameter of the Dirichlet distribution
    beta : float
        the beta parameter of the Dirichlet distribution
    id2word : gensim.corpora.Dictionary
        gensim dictionary
    doc_freq : iterable of list of (int, float)
        corpus in bag of words format
    corpus : List[str]
        corpus of documents

    Returns
    --------
    score : float
        the coherence score for the model
    """

    lda_model = LdaMulticore(
        corpus=doc_freq,
        id2word=id2word,
        num_topics=num_topics,
        alpha=alpha,
        eta=beta,
        random_state=42,
        chunksize=10000,
        passes=1,
        workers=7
    )
    
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=corpus,
        dictionary=id2word,
        coherence='c_v'
    )
    
    score = coherence_model.get_coherence()
    return score


def train():
    """
    Trains an LDA model. Outputs can be found at:
        - LDA Model: results/lda_model.pkl
        - Dictionary: results/id2word.pkl
        - BOW Corpus: results/doc_bow.pkl
    """

    # LOADING PROCESSED DATA
    with open('../data/processed_data.pkl', 'rb') as file:
        df = pickle.load(file)
        corpus = df['clean_lyrics'].to_list()


    # BIGRAMS
    bigram_transformer = phrases.Phrases(corpus, connector_words=phrases.ENGLISH_CONNECTOR_WORDS)
    corpus = list(bigram_transformer[corpus])


    # CREATING DICTIONARY
    id2word = corpora.Dictionary(corpus)
    id2word.filter_extremes(no_below=20, no_above=0.75, keep_n=None)
    doc_freq = [id2word.doc2bow(doc) for doc in corpus]


    # HYPERPARAMETER TUNING
    # grid search
    params = {
        'num_topics' : [_ for _ in range(4, 9)],
        'alpha' : [0.01, 0.05, 0.1, 0.5, 1, 3, 'symmetric', 'asymmetric'],
        'beta' : [0.01, 0.05, 0.1, 0.5, 1, 3, 'symmetric', 'auto'],
    }
    combos = product(params['num_topics'], params['alpha'], params['beta'])
    num_iters = np.prod([len(params[key]) for key in params.keys()])

    res = defaultdict(list)

    for (num_topics, alpha, beta) in tqdm(combos, total=num_iters):
            res['num_topics'].append(num_topics)
            res['alpha'].append(alpha)
            res['beta'].append(beta)
            
            coherence = calcCoherence(num_topics, alpha, beta, id2word, doc_freq, corpus)
            res['coherence'].append(coherence)
            
    df_cv = pd.DataFrame(res)
    df_cv.to_csv('../results/coherence_results.csv', index=False)

    # identifying best paramaters
    idx = np.argmax(df_cv['coherence'])
    best_params = df_cv.iloc[idx]
    num_topics, alpha, beta = best_params['num_topics'], best_params['alpha'], best_params['beta']


    # MODEL TRAINING
    lda_model = LdaMulticore(
        corpus=doc_freq,
        id2word=id2word,
        num_topics=num_topics,
        alpha=alpha,
        eta=beta,
        chunksize=2000,
        passes=10,
        per_word_topics=True
    )

    # SAVING ARTIFACTS
    with open('../models/lda_model.pkl', 'wb') as model_file:
        pickle.dump(lda_model, model_file)

    with open('../models/id2word.pkl', 'wb') as dict_file:
        pickle.dump(id2word, dict_file)

    with open('../models/doc_bow.pkl', 'wb') as bow_file:
        pickle.dump(doc_freq, bow_file)


if __name__ == '__main__':
     train()