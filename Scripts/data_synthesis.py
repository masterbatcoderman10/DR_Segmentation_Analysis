def synthesize_y(num_classes, max_score,batch_size=32):

    y_true = np.round(np.random.uniform(0, num_classes-1, (batch_size, 224, 224, 1)))
    y_true = tf.one_hot(y_true, depth=num_classes, axis=-1)
    y_true = tf.squeeze(y_true)
    y_pred = np.random.uniform(0, max_score, (batch_size, 224, 224, num_classes))
    #y_pred = np.zeros((224,224,3))
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    #y_pred = tf.one_hot(y_pred, depth=3, axis=-1)
    #y_pred = tf.squeeze(y_pred)