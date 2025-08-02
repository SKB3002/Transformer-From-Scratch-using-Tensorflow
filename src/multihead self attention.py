class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        #Splitting last dim into (num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        #Transopose for attention (batch, heads, seq_len, depth)
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self,v,k,q,mask= None):
        batch_size = tf.shape(q)[0]

        #1. Linear Projections
        q = self.wq(q)   #batch, seq_len, d_model
        k = self.wk(k)
        v = self.wv(v)

        #2. Split into heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        #3. Applying scaled dot product attention
        scaled_attention, attention_weights = ScaledDotProductAttention()(q,k,v,mask)

        #4. Combining Heads: Transpose + Reshape
        scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.num_heads * self.depth))

        #5. Final Linear Layer
        output = self.dense(concat_attention)
        return output, attention_weights
