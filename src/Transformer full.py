class Transformer(tf.keras.Model):
    def __init__(self, num_layers,d_model,num_heads,dff,
                input_vocab_size,target_vocab_size,
                pe_input,pe_target,rate=0.1):
        super().__init__()

        # Encoder & Decoder
        self.encoder = FullEncoder(
            num_layers,d_model,num_heads,dff,
            vocab_size=input_vocab_size,
            max_seq_len = pe_input,
            rate=rate
        )

        self.decoder = FullDecoder(
            num_layers, d_model, num_heads, dff,
            target_vocab_size = target_vocab_size,
            max_seq_len= pe_target,
            rate=rate
        )

        #Final Linear Projection
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self,inp,tar,training,
            enc_padding_mask=None,
            look_ahead_mask=None,
            dec_padding_mask=None):

        #Encoder
        enc_output, _ = self.encoder(inp,training=training,mask=enc_padding_mask)

        #Decoder
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask
        )


        #dec_output: (batch, tar_seq_len, d_model)


        #Final Dense Layer
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
