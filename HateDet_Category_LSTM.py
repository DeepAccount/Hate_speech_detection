from keras.layers import Embedding
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os.path

import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os.path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report


def run_lstm_hate_detection(df, output_folder):
    working_dir = os.getcwd() + '/' + output_folder + '/'
    modelfname = working_dir + "Model_LSTM_Category.mdl"

    df = df.dropna()
    x_texts = df[0]
    Y = df[2]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)

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
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Y = y.values
    X_train, test_x, Y_train, test_y = train_test_split(X, Y, test_size=0.15, shuffle=True, random_state=2)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=2)

    y_pred = model.predict(test_x)
    y_pred_decoded = np.zeros((y_pred.shape[0], 1))
    for r in range(y_pred_decoded.shape[0]):
        y_pred_decoded[r, 0] = np.argmax(y_pred[r, :])

    test_y_decoded = np.zeros((y_pred.shape[0], 1))
    for r in range(y_pred_decoded.shape[0]):
        test_y_decoded[r, 0] = np.argmax(test_y[r, :])

    result = test_y_decoded == y_pred_decoded
    print("Accuracy = " + str(np.count_nonzero(result == True) / result.shape[0]))

    print(classification_report(test_y_decoded, y_pred_decoded))
    joblib.dump(model, modelfname)
