class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self,vocab_size,d_model,max_seq_len):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size,d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.d_model = d_model

    def call(self,x):
        #Scaling Embeddings by SQRT(d_model)
        x = self.embedding(x) 
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))

        #Adding Positional Encoding
        x = self.pos_encoding(x) 

        return x
