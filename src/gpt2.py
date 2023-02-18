import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

class gpt2:

    # parameters
    models_dir:str = ""
    model_name:str = ""
    temperature:float = 0.7
    top_k:int = 40
    top_p:int = 1
    length:int = 80

    # private variables
    __enc__ = None # encoder/decoder for encoding/decoding out of tokens
    __sess__ = None # the model
    __context__ = None
    __output__ = None
    
    def load(self):

        self.__enc__ = encoder.get_encoder(self.model_name, self.models_dir)
        hparams = model.default_hparams()

        f = open(os.path.join(self.models_dir, self.model_name, 'hparams.json'))
        hparams.override_from_dict(json.load(f))
        f.close()

        self.__sess__ = tf.Session()
        self.__context__ = tf.placeholder(tf.int32, [1, None])
        self.__output__ = sample.sample_sequence(hparams=hparams, length=self.length, context=self.__context__, batch_size=1, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(self.models_dir, self.model_name))
        saver.restore(self.__sess__, ckpt)

    def prompt(self, prompt:str) -> str:
        context_tokens = self.__enc__.encode(prompt)
        out = self.__sess__.run(self.__output__, feed_dict={self.__context__: [context_tokens]})[:, len(context_tokens):]
        text = self.__enc__.decode(out[0])
        return text


gen = gpt2()
gen.models_dir = r"C:\Users\Tim\Downloads\gpt-2\models"
gen.model_name = "345M"
gen.load()

print("Loaded!")

response = gen.prompt("One effective way to improve time management is ")
print(response)