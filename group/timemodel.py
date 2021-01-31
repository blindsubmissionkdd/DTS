
import tensorflow as tf
import numpy as np
def dump_embedding(proto_embed, sample_embed, labels, dump_file='./plot/embeddings.txt'):
    proto_embed = tf.identity(proto_embed).numpy()
    sample_embed = tf.identity(sample_embed).numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             tf.squeeze(labels).numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')

def output_conv_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    assert d == y.shape[1]

    x = tf.broadcast_to(tf.expand_dims(x, axis=1), [n,m,d])
    y = tf.broadcast_to(tf.expand_dims(y, axis=0), [n,m,d])

    return tf.reduce_sum(tf.pow(x - y, 2),2)

class TAPNet(tf.keras.layers.Layer):
    """ Wrap VRNNCell into a RNN """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nfeat = 9
        self.ts_length = 128
        self.lstm_dim = 32

        self.nclass = 32
        self.dropout = 0
        self.use_metric = False
        self.use_lstm = True
        self.use_cnn = True
        self.use_att=True
        # parameters for random projection
        self.use_rp = True
        self.rp_group, self.rp_dim = 3, 6
        self.dilation = 1
        self.filters=[256,256,128]
        self.kernels = [8,5,3]
        self.layers = [300,100]
        self.metric_param =0.01



    def build(self, input_shape):
        # LSTM
        self.lstm = tf.keras.layers.LSTM(self.lstm_dim, return_sequences=True, return_state=True) #whole_seq_output, final_memory_state, final_carry_state
        # self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

        self.create_layers()

        # compute the size of input for fully connected layers
        fc_input = 0
        if self.use_cnn:
            conv_size = self.ts_length
            for i in range(len(self.filters)):
                conv_size = output_conv_size(conv_size, self.kernels[i], 1, 0)
            fc_input += conv_size
            # * filters[-1]
        if self.use_lstm:
            fc_input += conv_size * self.lstm_dim

        if self.use_rp:
            fc_input = self.rp_group * self.filters[2] + self.lstm_dim

        # Representation mapping function
        layers = [fc_input] + self.layers
        print("Layers", layers)

        # Optionally, the first layer can receive an `input_shape` argument:
        self.mapping = tf.keras.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add(tf.keras.layers.Dense(layers[i+1]))
            self.mapping.add(tf.keras.layers.BatchNormalization())
            self.mapping.add(tf.keras.layers.LeakyReLU())

        self.mapping.add(tf.keras.layers.Dense(layers[-1]))

        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add(tf.keras.layers.BatchNormalization())

        # Attention
        att_dim, semi_att_dim = 128, 128
        if self.use_att:
            self.att_models = []
            for _ in range(self.nclass):
                att_model = tf.keras.Sequential(
                    [tf.keras.layers.Dense(att_dim,activation="tanh"),
                    tf.keras.layers.Dense(1)]
                )
                self.att_models.append(att_model)

    def create_layers(self):
        self.conv_1_models = []
        self.idx = []
        if self.use_rp:
            for i in range(self.rp_group):
                self.conv_1_models.append(tf.keras.layers.Conv1D(filters=self.filters[0], kernel_size=self.kernels[0], strides=1, padding="same", dilation_rate=1))
                self.idx.append(np.random.permutation(self.nfeat)[0:self.rp_dim])

        else:
            self.conv_1 = tf.keras.layers.Conv1D(filters=self.filters[0], kernel_size=self.kernels[0], strides=1, padding="same", dilation_rate=1)
        self.conv_bn_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 =tf.keras.layers.Conv1D(filters=self.filters[1], kernel_size=self.kernels[1], strides=1, padding="same", dilation_rate=1)

        self.conv_bn_2 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv1D(filters=self.filters[2], kernel_size=self.kernels[2], strides=1, padding="same", dilation_rate=1)

        self.conv_bn_3 = tf.keras.layers.BatchNormalization()


    def call(self, inputs, labels, **kwargs):

        N = inputs.shape[0]
        if self.use_lstm:
            x_lstm = self.lstm(inputs)[0]
            x_lstm = tf.reduce_mean(x_lstm, axis=1)
            x_lstm = tf.reshape(x_lstm, [N, -1])

        inputs = tf.transpose(inputs, [0, 2, 1])
        if self.use_cnn:
            # Covolutional Network
            # input ts: # N * C * L
            if self.use_rp:
                for i in range(len(self.conv_1_models)):
                    # x_conv = x
                    # tmp = tf.gather(inputs, self.idx[i], axis=2)
                    x_conv = self.conv_1_models[i](tf.gather(inputs, self.idx[i], axis=1))
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = tf.nn.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = tf.nn.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = tf.nn.leaky_relu(x_conv)

                    x_conv = tf.reduce_mean(x_conv, axis=1)

                    if i == 0:
                        x_conv_sum = x_conv
                    else:
                        x_conv_sum = tf.concat([x_conv_sum, x_conv], axis=1)

                x_conv = x_conv_sum
            else:
                x_conv = inputs
                x_conv = self.conv_1(x_conv)  # N * C * L
                x_conv = self.conv_bn_1(x_conv)
                x_conv = tf.nn.leaky_relu(x_conv)

                x_conv = self.conv_2(x_conv)
                x_conv = self.conv_bn_2(x_conv)
                x_conv = tf.nn.leaky_relu(x_conv)

                x_conv = self.conv_3(x_conv)
                x_conv = self.conv_bn_3(x_conv)
                x_conv = tf.nn.leaky_relu(x_conv)

                x_conv = tf.reshape(x_conv, [N, -1])

        if self.use_lstm and self.use_cnn:
            x = tf.concat([x_conv, x_lstm], axis=1)
        elif self.use_lstm:
            x = x_lstm
        elif self.use_cnn:
            x = x_conv


        # linear mapping to low-dimensional space
        x = self.mapping(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx =  tf.squeeze(tf.where((tf.squeeze(labels)==i)!=False),axis=1)
            if self.use_att:
                A = self.att_models[i](tf.gather(x, idx, axis=0))  # N_k * 1
                A = tf.transpose(A)  # 1 * N_k
                A = tf.nn.softmax(A, axis=1)  # softmax over N_k

                class_repr = tf.matmul(A, tf.gather(x, idx, axis=0)) # 1 * L
                class_repr = tf.transpose(class_repr, [1, 0])  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr =tf.reduce_mean(tf.gather(x, idx, axis=0),axis=0)  # L * 1
            proto_list.append(tf.reshape(class_repr, [1, -1]))
        x_proto = tf.concat(proto_list, axis=0)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        proto_dists = tf.math.exp(-0.5*proto_dists)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = tf.reduce_sum(proto_dists) / num_proto_pairs

        dists = euclidean_dist(x, x_proto)

        dump_embedding(x_proto, x, labels)
        return tf.math.exp(-0.5*dists), proto_dist
        # return tf.math.exp(-0.5*dists)