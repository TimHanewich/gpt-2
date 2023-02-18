import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

# Model location and selection
model_name = "345M"
models_dir = r"C:\Users\Tim\Downloads\gpt-2\models"

# parameters
temperature = 0.7
top_k = 40
top_p = 1
length = 80 # output length, in tokens

enc = encoder.get_encoder(model_name, models_dir)
hparams = model.default_hparams()
with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

with tf.Session(graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [1, None])
    output = sample.sample_sequence(hparams=hparams, length=length, context=context, batch_size=1, temperature=temperature, top_k=top_k, top_p=top_p)

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver.restore(sess, ckpt)



    PROMPT = "A woman wakes up in a conference room with no memory of who or where she is. After being given a survey and learning she is Helly, a new hire at Lumon Industries, she is allowed to leave but finds she is unable to do so. She then sees a video explaining that she has undergone the severance procedure, which split her memories to create a version of herself that will only exist inside the workplace. Mark Scout, who works alongside Helly in Lumon's Macrodata Refinement (MDR) division, discovers he is being promoted to department head in light of coworker Petey's sudden departure. The outside version of Mark, a former history professor grieving his wife's death and living in the Lumon-subsidized town of Kier, encounters a man claiming to be Petey who gives him a letter with cryptic instructions. Mark returns home and interacts with his neighbor Mrs. Selvig, unaware that she is his boss, senior manager Harmony Cobel. Helly subjected herself to the severance procedure because "
    PROMPT = "Nice, France is a great place to visit because "
    PROMPT = "Dear friend, I wanted to write to you to tell you all about why I love dogs. Dogs are cute and cuddly. I love dogs because "
    PROMPT = "One effective way to improve time management is "
    
    


    context_tokens = enc.encode(PROMPT)
    
    out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
    text = enc.decode(out[0])
    print(text)