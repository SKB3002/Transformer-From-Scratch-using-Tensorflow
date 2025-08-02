def positional_encoding(max_seq_len, d_model):
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angles = pos * angle_rates

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    pos_encoding = angles[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class FullEncoder(tf.keras.layers.Layer):
    def __init__(self,num_layers, d_model, num_heads,dff,
               vocab_size,max_seq_len,rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        #Embedding Layers
        self.embedding = tf.keras.layers.Embedding(vocab_size,d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)


        self.enc_layers = [
            EncoderLayer(d_model,num_heads,dff,rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        seq_len = tf.shape(x)[1]

        #Token Embedding + Scaling + Positional Encoding
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        #Passing through N encoder layers
        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, training=training, mask=mask)

        return x, attn_weights
