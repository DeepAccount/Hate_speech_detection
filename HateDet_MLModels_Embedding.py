import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def run_embedding_hate_detection(df):
    """
    Runs multiple ML models using Glove word embedding and calculates accuracy metrics
    :param df: data
    :type df: pandas dataframe
    """
    x_texts = df[0]
    y = df[1]

    # python -m spacy download en_core_web_md
    nlp = spacy.load('en_core_web_md')

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

    train_x, test_x, train_y, test_y = train_test_split(x_texts_vector_input, y, random_state=2, stratify=y,
                                                        shuffle=True,
                                                        test_size=0.05)
    for mdl in ('Logistic Regression', 'RandomForest', 'SVC'):
        if mdl == 'Logistic Regression': model_hate = LogisticRegression(solver='lbfgs', max_iter=10000)
        if mdl == 'MultinomialNB': model_hate = MultinomialNB()
        if mdl == 'RandomForest': model_hate = RandomForestClassifier(n_estimators=10)
        if mdl == 'SVC': model_hate = SVC(kernel='linear', probability=True)

        print('######## ' + mdl + ' ########')
        model_hate.fit(train_x, train_y)
        y_pred = model_hate.predict(test_x)
        print('Accuracy Score = ' + str(model_hate.score(test_x, test_y)))
        cf_matrix = confusion_matrix(test_y, y_pred)
        print(cf_matrix)
        f1 = f1_score(test_y, y_pred)
        print("F1 Score " + str(f1))
