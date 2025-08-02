def point_wise_ffn(d_model,dff):  
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation = "relu"),
        tf.keras.layers.Dense(d_model)
    ])
