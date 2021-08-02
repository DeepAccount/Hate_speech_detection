from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import os.path
import numpy as np

from sklearn.model_selection import train_test_split

import joblib
import os.path

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def run_lstm_hate_detection(df, output_folder):
    """

    :param df:
    :type df:
    :param output_folder:
    :type output_folder:
    """
    df = df.drop_duplicates()
    df = df.dropna()

    working_dir = os.getcwd() + '/' + output_folder + '/'

    ################### Creating Model for Hate Detection ################
    LSTM_out_Size = 200
    tweet_emb_fname = "tweet_embedding_Categorization.npy"
    modelfname = working_dir + "Model_LSTM_Word_Embed_Categorization.mdl"
    x_texts = df[0]
    y = df[2]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)

    embedding_available = False
    # python -m spacy download en_core_web_md
    import en_core_web_sm

    nlp = en_core_web_sm.load()
    x_texts_vector_input = np.zeros((len(x_texts), 6720))

    if not (os.path.isfile(tweet_emb_fname)):
        print("Creating tweet embeddings...")
        x_texts_vector = []
        zero = np.zeros(96)
        ii = 0
        for x in x_texts:
            tweet_vec = np.zeros(0)
            doc = nlp(x)
            while (len(doc) > 70):
                x = x.rsplit(' ', 1)[0]
                doc = nlp(x)
            for i in range(70 - len(doc)):
                tweet_vec = np.append(tweet_vec, zero)

            for i in range(len(doc)):
                tweet_vec = np.append(tweet_vec, doc[i].vector)

            #        x_texts_vector.append(tweet_vec)
            x_texts_vector_input[ii, :] = tweet_vec
            ii += 1

        #    x_texts_vector_input = np.array(x_texts_vector)
        print("Saving tweet embeddings...")
        np.save(tweet_emb_fname, x_texts_vector_input)
    else:
        print("Loading tweet embeddings...")
        x_texts_vector_input = np.load(tweet_emb_fname)
        print("Data load done")

    X = x_texts_vector_input
    X = np.reshape(X, (X.shape[0], 70, 96))

    batch_size = 32

    model = Sequential()
    model.add(LSTM(LSTM_out_Size, activation='relu', input_shape=(70, 96), dropout_U=0.2, dropout_W=0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print("Splitting data set")
    X_train, test_x, Y_train, test_y = train_test_split(X, Y, test_size=0.15, shuffle=True, random_state=2)

    from sklearn.metrics import classification_report

    best_acc = 0

    if (os.path.isfile(modelfname)):
        print("Loaing Model from file...")
        model = joblib.load(modelfname)
        y_pred = model.predict(test_x)
        y_pred_decoded = np.zeros((y_pred.shape[0], 1))
        for r in range(y_pred_decoded.shape[0]):
            y_pred_decoded[r, 0] = np.argmax(y_pred[r, :])

        test_y_decoded = np.zeros((y_pred.shape[0], 1))
        for r in range(y_pred_decoded.shape[0]):
            test_y_decoded[r, 0] = np.argmax(test_y[r, :])

        print(classification_report(test_y_decoded, y_pred_decoded))

        result = test_y_decoded == y_pred_decoded
        acc = np.count_nonzero(result == True) / result.shape[0]
        print("Current Accuracy = " + str(acc))

        if acc > best_acc:
            best_acc = acc

    print("Model fitting...")
    for i in range(20):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
        y_pred = model.predict(test_x)

        y_pred_decoded = np.zeros((y_pred.shape[0], 1))
        for r in range(y_pred_decoded.shape[0]):
            y_pred_decoded[r, 0] = np.argmax(y_pred[r, :])

        test_y_decoded = np.zeros((y_pred.shape[0], 1))
        for r in range(y_pred_decoded.shape[0]):
            test_y_decoded[r, 0] = np.argmax(test_y[r, :])

        print(classification_report(test_y_decoded, y_pred_decoded))

        result = test_y_decoded == y_pred_decoded
        acc = np.count_nonzero(result == True) / result.shape[0]
        print("Current Accuracy = " + str(acc))

        if acc > best_acc:
            print("Saving the model...")
            joblib.dump(model, modelfname)
            best_acc = acc
        print("Best Accuracy = " + str(best_acc))
        print("")
