# Music Topic Modeling

Topic modeling of music across five genres: country, pop, rock, R&B, and hip-hop. 

## Data Description

Lists of artists corresponding to each genre were scraped using BeautifulSoup from the websites listed below. Complete discographies for each artist, excluding features, were then obtained via the Genius API for a total of 350k songs. Metadata collected with each song includes release date (year only), genre, and artist. 
- Country: [https://en.wikipedia.org/wiki/List_of_country_music_performers](https://en.wikipedia.org/wiki/List_of_country_music_performers)
- Rock: [https://en.wikipedia.org/wiki/List_of_hard_rock_musicians_(A%E2%80%93M)](https://en.wikipedia.org/wiki/List_of_hard_rock_musicians_(A%E2%80%93M))
- R&B: [https://en.wikipedia.org/wiki/List_of_R%26B_musicians](https://en.wikipedia.org/wiki/List_of_R%26B_musicians)
- Hip-Hop: [https://en.wikipedia.org/wiki/List_of_hip_hop_musicians](https://en.wikipedia.org/wiki/List_of_hip_hop_musicians)
- Pop: [https://today.yougov.com/ratings/entertainment/fame/pop-artists/all](https://today.yougov.com/ratings/entertainment/fame/pop-artists/all)

##  Methodology
### Data Collection and Preprocessing
Data was collected using BeautifulSoup coupled with the Genius API and uploaded to a PostgreSQL database using psycopg2. After data collection, song lyrics were preprocessed to prepare for topic modeling, which included:
- Language Detection (langdetect)
- Spell Check (symspellpy)
- Tokenization
- Lemmatization (NLTK)
- Bigram Detection (gensim)

The code for data collection and preprocessing can be found at src/scrape.py and src/preprocess.py.

### LDA
Topic modeling is an unsupervised approach for classification of documents, comparable to clustering algorithms for numeric data. Latent Dirichlet Allocation (LDA) is a widely used method for topic modeling that aims to find which topics a document belongs to based on the frequency of words that appear in it. The three primary hyperparameters that define an LDA model include the alpha and beta values that describe the Dirichlet distribution as well as the number of topics. In order to tune these hyperparameters, I ran a grid search with the scoring metric being  coherence score, which aims to assess how interpretable the topics are by measuring the degree of semantic similarity between high scoring words in each topic. The final model was then trained using the optimal values for these hyperparameters. The full code for training the model can be found at src/train.py.

## Results
LDA results were visualized using pyLDAvis and can be viewed here: [https://suhulavu.github.io/Music-Topic-Modeling/](https://suhulavu.github.io/Music-Topic-Modeling/)

To further analyze the results, dimensionality reduction via UMAP was applied to the matrix describing each song in topic space in order to view the data in two dimensions. Due to file size limits on Github, the interactive Altair chart could not be uploaded to Github pages, but a static plot can be viewed below:
![umap pic](/docs/umap_viz.png)
