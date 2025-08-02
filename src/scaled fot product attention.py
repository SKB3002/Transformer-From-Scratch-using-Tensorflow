class ScaledDotProductAttention(tf.keras.layers.Layer):
    def call(self, q, k, v, mask=None):
        #Compute similarity scores
        matmul_qk = tf.matmul(q,k,transpose_b = True)

        #Scale by SQRT
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = matmul_qk / tf.math.sqrt(dk)

        #Optional Mask
        if mask is not None:
            scaled_attention += (mask * -1e9)

        #Computing Attention weights
        attention_weights = tf.nn.softmax(scaled_attention, axis = -1)

        #Weighted sum of values
        output = tf.matmul(attention_weights, v)

        return output, attention_weights
