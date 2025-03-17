import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPool2D, Flatten, Dropout, GRU


# Add attention layer to the deep learning network
class attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        super(attention, self).build(input_shape)
 
    def call(self, x, **kwargs):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W))
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        return context, alpha


def create_gru_model(len_thre):
    # Define GRU model architecture
    model = Sequential()
    model.add(InputLayer(input_shape=(len_thre, 4), name="rnn_input"))
    model.add(GRU(128, dropout=0.2, return_sequences=True, name="rnn1"))
    model.add(GRU(64, dropout=0.2, return_sequences=True, name="rnn2"))
    model.add(Flatten(name="rnn_flatten"))
    model.add(Dropout(0.5, name="rnn_dropout1"))
    model.add(Dense(128, activation="relu", name="rnn_fc1"))
    return model

def create_cnn_model(k):
    model = Sequential()
    model.add(InputLayer(input_shape=(1, pow(4, k), 1), name="cnn_input"))
    model.add(Conv2D(filters=64, kernel_size=(1, 3), activation="relu", name="cnn_conv1"))
    model.add(MaxPool2D(pool_size=(1, 2), name="cnn_pool1"))
    model.add(Conv2D(filters=128, kernel_size=(1, 3),  activation="relu", name="cnn_conv2"))
    model.add(MaxPool2D(pool_size=(1, 2), name="cnn_pool2"))
    model.add(Conv2D(filters=256, kernel_size=(1, 3),  activation="relu", name="cnn_conv3"))
    model.add(MaxPool2D(pool_size=(1, 2), name="cnn_pool3"))
    model.add(Flatten(name="cnn_flatten"))
    model.add(Dropout(0.5, name="cnn_dropout1"))
    model.add(Dense(128, activation="relu", name="cnn_fc1"))
    return model
    

# Add an attention layer to the CNN and RNN output joint layer
def create_attn_model(kmer, len_thre, class_num):
    cnn_model = create_cnn_model(kmer)
    # rnn_model = create_rnn_model(len_thre)
    rnn_model = create_gru_model(len_thre)
    merge_layer = K.stack((cnn_model.output, rnn_model.output), axis=1)
    attn_layer, attention_weight = attention(name="attention")(merge_layer)
    attn_layer = K.sum(attn_layer, axis=1)
    dense_layer = Dense(128, activation="relu", name="fc")(attn_layer)
    dropout_layer = Dropout(0.5, name="dropout")(dense_layer)
    output_layer = Dense(int(class_num), activation="softmax", name="output")(dropout_layer)
    model = Model(inputs=[cnn_model.input, rnn_model.input], outputs=output_layer)
    attention_model = Model([cnn_model.input, rnn_model.input], attention_weight)
    return model, attention_model


def train_attn_model(X_oh_train, X_oh_test, X_kmer_train, X_kmer_test, y_train, y_test, col_name, k, l):
    class_num = len(set(col_name))
    model, attention_model = create_attn_model(k, l, class_num)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])
    
    X_oh_train, X_kmer_train, y_train = shuffle(X_oh_train, X_kmer_train, y_train)
    model.fit([X_kmer_train, X_oh_train], y_train, batch_size=32, epochs=10, verbose=2, validation_data=([X_kmer_test, X_oh_test], y_test))

    return model
