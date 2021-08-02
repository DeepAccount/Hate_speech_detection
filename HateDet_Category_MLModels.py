import os.path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def run_ml_models_hate_detection(df, output_folder):
    """

    Runs multiple ML models and calculates accuracy and F1 score,
    Used for finding category
    :param df: data for processing
    :type df: pandas dataframe
    :param output_folder: folder to store interim information
    :type output_folder: string
    """
    working_dir = os.getcwd() + '/' + output_folder + '/'

    print("#######################################")
    print('Creating Hate Category detection Model...')
    category = 'Category'
    df = df.drop_duplicates()
    df = df[df[1] == 1]
    df = df.dropna()
    x_texts = df[0]
    y = df[2]

    model_hate = ''
    for vect in ("CountVectorizer", "TfidfVectorizer"):
        if vect == "CountVectorizer":  vectorizer = CountVectorizer(ngram_range=(1, 3))
        if vect == "TfidfVectorizer": vectorizer = TfidfVectorizer(max_features=None)
        vectorizerfname = working_dir + vect + category + '.vec'
        if (os.path.isfile(vectorizerfname)):
            vectorizer = joblib.load(vectorizerfname)
        else:
            vectorizer.fit(x_texts)
            joblib.dump(vectorizer, vectorizerfname)
        x_Texts_features = vectorizer.transform(x_texts)
        train_x, test_x, train_y, test_y = train_test_split(x_Texts_features, y, random_state=2,
                                                            shuffle=True,
                                                            test_size=0.05)

        for mdl in ('Logistic Regression', 'MultinomialNB', 'RandomForest', 'SVC'):
            if mdl == 'Logistic Regression': model_hate = LogisticRegression(multi_class='ovr', solver='lbfgs')
            if mdl == 'MultinomialNB': model_hate = MultinomialNB()
            if mdl == 'RandomForest': model_hate = RandomForestClassifier(n_estimators=10)
            if mdl == 'SVC': model_hate = SVC(kernel='linear', probability=True)

            print('######## ' + mdl + " " + vect + ' ########')
            modelfname = mdl + " " + vect + '.mdl'
            modelfname = working_dir + modelfname + category
            if (os.path.isfile(modelfname)):
                model_hate = joblib.load(modelfname)
            else:
                model_hate.fit(train_x, train_y)
                joblib.dump(model_hate, modelfname)
            y_pred = model_hate.predict(test_x)
            print('Accuracy Score = ' + str(model_hate.score(test_x, test_y)))
            # cf_matrix = confusion_matrix(test_y, y_pred)
            # print(cf_matrix)
