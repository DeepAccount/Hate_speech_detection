from keras.layers import Embedding
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import joblib
import os



def run_lstm_hate_detection(df, output_folder):
    """
    Hate/Non hate claffication using LSTM
    :param df: data
    :type df: Pandas dataframe
    """
    working_dir = os.getcwd() + '/' + output_folder + '/'
    modelfname = working_dir + "Model_LSTM.mdl"
    df = df.drop_duplicates()

    x_texts = df[0]
    Y = df[1].values

    tokenizer = Tokenizer(num_words=2500, lower=True, split=' ')
    tokenizer.fit_on_texts(x_texts.values)
    X = tokenizer.texts_to_sequences(x_texts.values)
    X = pad_sequences(X, maxlen=70)

    embed_dim = 128
    lstm_out = 300
    batch_size = 32

    model = Sequential()
    model.add(Embedding(2500, embed_dim, input_length=X.shape[1], dropout=0.2))
    model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    X_train, test_x, Y_train, test_y = train_test_split(X, Y, test_size=0.15, shuffle=True, random_state=2)

    for i in range(5):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)

        y_pred = model.predict(test_x)
        y = y_pred > 0.5
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

        print("Current F1 score = " + str(f1_score(test_y, y)))
        print("Current Precision score = " + str(precision_score(test_y, y)))
        print("Current Recall score = " + str(recall_score(test_y, y)))
        print("")

    joblib.dump(model, modelfname)
