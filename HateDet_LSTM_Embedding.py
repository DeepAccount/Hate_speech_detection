from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os.path


def run_lstm_hate_detection(df, output_folder):
    """

    :param df:
    :type df:
    :param output_folder:
    :type output_folder:
    """
    working_dir = os.getcwd() + '/' + output_folder + '/'

    LSTM_out_Size = 200
    tweet_emb_fname = "tweet_embedding_classification.npy"
    modelfname = working_dir + "Model_LSTM_Word_Embed.mdl"
    x_texts = df[0]
    y = df[1]
    embedding_available = False
    # python -m spacy download en_core_web_md
    import en_core_web_sm

    nlp = en_core_web_sm.load()

    if not (os.path.isfile(tweet_emb_fname)):
        x_texts_vector = []
        zero = np.zeros(96)
        for x in x_texts:
            tweet_vec = np.zeros(0)
            doc = nlp(x)

            for i in range(70 - len(doc)):
                tweet_vec = np.append(tweet_vec, zero)

            for i in range(len(doc)):
                tweet_vec = np.append(tweet_vec, doc[i].vector)

            x_texts_vector.append(tweet_vec)

        x_texts_vector_input = np.array(x_texts_vector)
        np.save(tweet_emb_fname, x_texts_vector_input)
    else:
        x_texts_vector_input = np.load(tweet_emb_fname)
        print("Data load done")

    X = x_texts_vector_input
    X = np.reshape(X, (X.shape[0], 70, 96))
    Y = y.values

    batch_size = 32

    model = Sequential()
    model.add(LSTM(LSTM_out_Size, activation='relu', input_shape=(70, 96), dropout_U=0.2, dropout_W=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print("Splitting data set")
    X_train, test_x, Y_train, test_y = train_test_split(X, Y, test_size=0.15, shuffle=True, random_state=2)

    from sklearn.metrics import classification_report
    best_f1_score = 0

    if (os.path.isfile(modelfname)):
        print("Loaing Model from file...")
        model = joblib.load(modelfname)
        y_pred = model.predict(test_x)
        y = y_pred > 0.5
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        f1 = (f1_score(test_y, y))
        print("Current F1 score = " + str(f1))
        print("Current Precision score = " + str(precision_score(test_y, y)))
        print("Current Recall score = " + str(recall_score(test_y, y)))
        print(classification_report(test_y, y))
        print("")
        if f1 > best_f1_score:
            best_f1_score = f1

    print("Model fitting...")
    for i in range(10):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)

        y_pred = model.predict(test_x)
        y = y_pred > 0.5

        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        f1 = (f1_score(test_y, y))
        print("Current F1 score = " + str(f1))
        print("Current Precision score = " + str(precision_score(test_y, y)))
        print("Current Recall score = " + str(recall_score(test_y, y)))
        print(classification_report(test_y, y))

        if f1 > best_f1_score:
            joblib.dump(model, modelfname)
            best_f1_score = f1
        print("Best score = " + str(best_f1_score))
