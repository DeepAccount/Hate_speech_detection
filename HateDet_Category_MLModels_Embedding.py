import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def run_embedding_hate_detection(df):
    """

    Runs multiple ML models using Glove word embedding and calculates accuracy metrics
    used for finding category
    :param df: data
    :type df: pandas dataframe:
    """

    print("#######################################")
    print('Creating Hate Category detection Model...')
    category = 'Category'
    df = df.drop_duplicates()
    df = df[df[1] == 1]
    df = df.dropna()
    x_texts = df[0]
    y = df[2]
    print(x_texts.shape)

    nlp = spacy.load('en_core_web_md')

    saved_file_name = "tweet_embedding_category-96.npy"
    try:
        x_texts_vector_input = np.load(saved_file_name)
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
        np.save('tweet_embedding_category-96', x_texts_vector_input)

    print("Embeddings Loaded")
    print(x_texts_vector_input.shape)
    print(y.shape)
    train_x, test_x, train_y, test_y = train_test_split(x_texts_vector_input, y, random_state=2,
                                                        shuffle=True,
                                                        test_size=0.05)

    for mdl in ('Logistic Regression', 'RandomForest', 'SVC'):
        if mdl == 'Logistic Regression': model_hate = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=10000 )
        if mdl == 'RandomForest': model_hate = RandomForestClassifier(n_estimators=10)
        if mdl == 'SVC': model_hate = SVC(kernel='linear', probability=True)

        print('######## ' + mdl + " " + "Category" + ' ########')
        model_hate.fit(train_x, train_y)
        print('Accuracy Score = ' + str(model_hate.score(test_x, test_y)))
        # cf_matrix = confusion_matrix(test_y, y_pred)
        # print(cf_matrix)
