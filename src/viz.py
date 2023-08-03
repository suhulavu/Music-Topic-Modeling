# IMPORTS
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pickle
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import numpy as np
from tqdm import tqdm
import altair as alt
from umap import UMAP
import argparse


def genViz(n_neighbors, min_dist):
    """
    Renders two visualizations of LDA results:
        - pyLDAvis viz: results/lda_viz.html
        - UMAP viz: results/umap_viz.png
    
    Parameters
    -----------
    n_neighbors : int
        UMAP parameter
    min_dist : float
        UMAP parameter
    """

    # LOADING DATA/MODELS
    with open('../data/processed_data.pkl', 'rb') as file:
        df = pickle.load(file)
    
    with open('../models/id2word.pkl', 'rb') as dict_file:
        id2word = pickle.load(dict_file)

    with open('../models/doc_bow.pkl', 'rb') as bow_file:
        doc_bow = pickle.load(bow_file)

    with open('../models/lda_model.pkl', 'rb') as model_file:
        lda_model = pickle.load(model_file)

    
    # LDA VISUALIZATION
    vis = pyLDAvis.gensim_models.prepare(lda_model, doc_bow, id2word)
    pyLDAvis.save_html(vis, '../results/lda_viz.html')


    # INTERACTIVE VISUALIZATION

    # calculating topic distributions
    topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in tqdm(doc_bow, desc='Calculating Topic Distributions')]
    num_topics = len(topic_distributions[0])
    topic_vecs = np.zeros((df.shape[0], num_topics))
    for i, doc_topics in enumerate(topic_distributions):
        dist = [x[1] for x in doc_topics]
        topic_vecs[i, :] = dist

    # creating final dataframe for visualization
    df_topic = pd.DataFrame(topic_vecs, columns=['{}'.format(i) for i in range(1, num_topics + 1)])
    df_final = pd.concat([df.reset_index(drop=True), df_topic.reset_index(drop=True)], axis=1).drop(columns=['lyrics', 'clean_lyrics', 'id'])
    with open('../results/topic_df.pkl', 'wb') as file:
        pickle.dump(df_final, file)

    # dimensionality reduction
    umap = UMAP(random_state=42, n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    data = df_topic.to_numpy()
    reduced_data = umap.fit_transform(data)
    df_final[['PC_1', 'PC_2']] = reduced_data

    # altair viz
    df_final['tooltip'] = 'Song: ' + df_final['song'] + ' | Artist: ' + df_final['artist'] + ' | Year: ' + df_final['year'].astype(str)
    df_final = df_final.loc[df_final['year'] > 1]
    df_final['year'] = pd.to_datetime(df_final['year'])
    alt.data_transformers.disable_max_rows()

    single = alt.selection_point(on='mouseover', nearest=False)

    chart = alt.Chart(df).mark_point().encode(
        x=alt.X('PC_1:Q'),
        y=alt.Y('PC_2:Q'),
        tooltip='tooltip:N',
        color=alt.condition(single, 'year:T', alt.value('lightgray')),
        shape='genre:N'
    ).interactive(
    ).add_params(
        single
    ).properties(
        width='container',
        height=800
    )

    chart.save('../results/umap_viz.html')
    chart.save('../results/umap_viz.png')



if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser(description='Visualize LDA Results')
    parser.add_argument('--n_neighbors', dest='n_neighbors', type=int, default=15, action='store', help='UMAP n_neighbors param')
    parser.add_argument('--min_dist', dest='min_dist', action='store', default=0.1, type=float, help='UMAP min_dist param')
    args = parser.parse_args()

    # generate visualizations
    genViz(args.n_neighbors, args.min_dist)