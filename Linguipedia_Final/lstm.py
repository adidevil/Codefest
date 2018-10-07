import numpy as np
import sys
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Permute, Reshape, Lambda, RepeatVector, merge, Flatten, multiply
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
from keras import backend as K
import utils
from keras.preprocessing.sequence import pad_sequences

FREQ_DIST_FILE = 'train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = 'train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = 'train-processed.csv'
TEST_PROCESSED_FILE = 'test-processed.csv'
GLOVE_FILE = '../Linguipedia/twitter-sentiment-analysis/dataset/glove-seeds.txt'
dim = 200


def get_glove_vectors(vocab):
    print 'Looking for GLOVE vectors'
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r') as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 0)
            tokens = line.split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print '\n'
    print 'Found %d words in GLOVE' % found
    return glove_vectors


def get_feature_vector(tweet):
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(csv_file, test_file=True):
    tweets = []
    labels = []
    print 'Generating feature vectors'
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append(feature_vector)
            else:
                tweets.append(feature_vector)
                labels.append(int(sentiment))
            utils.write_status(i + 1, total)
    print '\n'
    return tweets, np.array(labels)


def attention_3d_block(inputs, TIME_STEPS):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def model_attention_applied_after_lstm(vocab_size, dim, embedding_matrix, max_length):
    inputs = Input(shape=(max_length,))
    inputs1 = Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length)(inputs)
    inputs1 = Dropout(0.5)(inputs1)
    lstm_out = LSTM(128, return_sequences=True)(inputs1)
    attention_mul = attention_3d_block(lstm_out, max_length)
    attention_mul = Flatten()(attention_mul)
    output_pre = Dense(64, activation='relu')(attention_mul)
    output_drop = Dropout(0.5)(output_pre)
    output = Dense(1, activation='sigmoid')(output_drop)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':
    train = len(sys.argv) == 1
    np.random.seed(1337)
    vocab_size = 90000
    batch_size = 500
    max_length = 40
    vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)
    glove_vectors = get_glove_vectors(vocab)
    tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01
    for word, i in vocab.items():
        glove_vector = glove_vectors.get(word)
        if glove_vector is not None:
            embedding_matrix[i] = glove_vector
    tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
    shuffled_indices = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled_indices]
    labels = labels[shuffled_indices]
    if train:
        model = model_attention_applied_after_lstm(vocab_size, dim, embedding_matrix, max_length)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        filepath = "./models/lstm-{epoch:02d}-{loss:0.3f}-{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
        print model.summary()
        model.fit(tweets, labels, batch_size=128, epochs=15, validation_split=0.1, shuffle=True, callbacks=[checkpoint, reduce_lr])
    else:
        model = load_model(sys.argv[1])
        print model.summary()
        test_tweets, _ = process_tweets(TEST_PROCESSED_FILE, test_file=True)
        test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
        predictions = model.predict(test_tweets, batch_size=128, verbose=1)
        results = zip(map(str, range(len(test_tweets))), np.round(predictions[:, 0]).astype(int))
        utils.save_results_to_csv(results, 'attention_lstm.csv')
