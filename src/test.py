import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

model_name = "117M"
models_dir = r"C:\Users\Tim\Downloads\gpt-2\models"

enc = encoder.get_encoder(model_name, models_dir)
hparams = model.default_hparams()
with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

length = hparams.n_ctx // 2

with tf.Session(graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [1, None])
    np.random.seed(None)
    tf.set_random_seed(None)
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=1,
        temperature=0.7, top_k=40, top_p=1
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver.restore(sess, ckpt)





    PROMPT = "My favorite operating system is "
    context_tokens = enc.encode(PROMPT)
    
    out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
    text = enc.decode(out[0])
    print(text)