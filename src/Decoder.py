class FullDecoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,
                target_vocab_size,max_seq_len,rate = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers =num_layers

        #Token Embedding + Positional Encoding
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self,x,enc_output,training,look_ahead_mask = None, padding_mask = None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        #Token emb + Scaling + Pos Enc
        x = self.embeddong = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training = training)

        # Stack Decoder layer
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, 
                training = training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # where x : batch, target_seq_len, d_model
        return x, attention_weights
