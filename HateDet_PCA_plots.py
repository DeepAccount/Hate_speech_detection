import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def draw_pca_plot(df):
    """
    Creates PCA plots for different types input vector options
    :param df: data
    :type df: pandas dataframe
    """
    x_texts = df[0]
    y = df[1]

    multiple_types = ['TF-IDF', 'Word Embedding - 300 dim', 'Word Embedding - 96 dim']
    for vector_type in multiple_types:
        x_texts_vector_input = get_vector(x_texts, vector_type)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x_texts_vector_input)
        print("done")
        data1 = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        data = data1.values

        label_0_x = []
        label_1_x = []
        label_0_y = []
        label_1_y = []

        cntr = 0
        for label in y:
            if label == 0:
                label_0_x.append(data[cntr][0])
                label_0_y.append(data[cntr][1])
            if label == 1:
                label_1_x.append(data[cntr][0])
                label_1_y.append(data[cntr][1])
            cntr = cntr + 1

        style.use("ggplot")

        plt.scatter(label_0_x, label_0_y, c='g', label="Non-Hate", s=4)
        plt.scatter(label_1_x, label_1_y, c='r', label="Hate", s=4)

        plt.title("PCA PLOT HATE/Non-Hate " + vector_type)
        plt.ylabel("PC2")
        plt.xlabel("PC1")
        plt.legend()
        plt.show()


def get_vector(x_texts, vectorization_type):
    """

    :param x_texts: data
    :type x_texts: pandas dataframe
    :param vectorization_type: type of vectorization to be plotted
    :type vectorization_type: String
    :return: data
    :rtype: pandas dataframe
    """
    if vectorization_type == 'TF-IDF':
        vectorizer = TfidfVectorizer(max_features=None)
        vectorizer.fit(x_texts)
        x_Texts_features = vectorizer.transform(x_texts)
        return x_Texts_features.toarray()

    if vectorization_type == 'Word Embedding - 300 dim':
        saved_file_name = "tweet_embedding.npy"
        try:
            x_texts_vector_input = np.load(saved_file_name)
            return x_texts_vector_input
        except:
            # python -m spacy download en_core_web_sm
            nlp = spacy.load('en_core_web_md')

            x_texts_vector = []
            zero = np.zeros(300)
            for x in x_texts:
                tweet_vec = np.zeros(0)
                doc = nlp(x)
                for i in range(50):
                    if i > len(doc) - 1:
                        tweet_vec = np.append(tweet_vec, zero)
                    else:
                        tweet_vec = np.append(tweet_vec, doc[i].vector)
                x_texts_vector.append(tweet_vec)
            x_texts_vector_input = np.array(x_texts_vector)
            np.save('tweet_embedding', x_texts_vector_input)
            return x_texts_vector_input

    if vectorization_type == 'Word Embedding - 96 dim':
        saved_file_name = "tweet_embedding-96.npy"
        try:
            x_texts_vector_input = np.load(saved_file_name)
            return x_texts_vector_input
        except:

            # python -m spacy download en_core_web_sm
            nlp = spacy.load('en_core_web_sm')

            x_texts_vector = []
            zero = np.zeros(96)
            for x in x_texts:
                tweet_vec = np.zeros(0)
                doc = nlp(x)
                for i in range(50):
                    if i > len(doc) - 1:
                        tweet_vec = np.append(tweet_vec, zero)
                    else:
                        tweet_vec = np.append(tweet_vec, doc[i].vector)
                x_texts_vector.append(tweet_vec)
            x_texts_vector_input = np.array(x_texts_vector)
            np.save('tweet_embedding', x_texts_vector_input)
            return x_texts_vector_input
