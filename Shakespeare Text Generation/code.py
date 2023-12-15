import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])
    
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Uncomment the below commented code and execute only once to create and save the model 'textgenerator.model'
# Execute the remainder of the code only after running the below commented code
'''model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

model.fit(x,y, batch_size=256, epochs=4)

model.save('textgenerator.model') '''

model = tf.keras.models.load_model('textgenerator.model')

def sample(preds, temperature=0.1):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds)/temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_text(length, temperature):
  start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
  generated = ''
  sentence = text[start_index: start_index + SEQ_LENGTH]
  generated += sentence

  for i in range(length):
    x = np.zeros((1, SEQ_LENGTH, len(characters)))

    for t, character in enumerate(sentence):
      x[0, t, char_to_index[character]] = 1

    predictions = model.predict(x, verbose=0)[0]
    next_index = sample(predictions, temperature)
    next_character = index_to_char[next_index]

    generated += next_character
    sentence = sentence[1:] + next_character

  return generated

print('-----------0.2-----------')
print(generate_text(300, 0.2))
print('-----------0.4-----------')
print(generate_text(300, 0.4))
print('-----------0.6-----------')
print(generate_text(300, 0.6))
print('-----------0.8-----------')
print(generate_text(300, 0.8))
print('-----------1-----------')
print(generate_text(300, 1.0))

'''
=========OUTPUT==========

-----------0.2-----------
 nothing she does or seems
but smacks of the mest the the theres and the heres and the the the that the the the to the sour that sout the seand the mand the that the came the thes and the paren the courst the will the mand the mand the meath the fores the manger the sand the sour the beath the beather the the mersen the dount the course t
-----------0.4-----------
ill he be eased
with being nothing. musing his the mere of comene.

kenger:
seest the cour stinker ond and thase that ous the the to the then that hall his mears the mist fore, the hand your the pering,
that and heres of in the wise that sore belers thit spall the ther dones and and the came so me the the lame that the to the to the to th
-----------0.6-----------
t.
you edward, shall unto my lord cobhame prathe and is dall famed me the fore, of erso theirs and wores, and with the corees, and chave the coursis and but and fort, so fath the thin.

hichirs thing and hay dore thar ward of om shall o have of the fore thourere sourdsen dore forounest ane chanden her stound in what the mach lave as ie fo
-----------0.8-----------
prettiest babe that e'er i nursed:
an i hen geace.

toriengrower:
sith the marter soll sprokes,
apdety of bead sour hay some,
thy ligsty of to thes encene hen teates th chyorso his will; brad ow brtath this of mast of are ftarmee:
ne hie the horke at thes prawifristm magus and and har.

badog, in thills for brepchart,
for wancerethis mist
-----------1-----------
coal-black hair
shall, whiles thy head in forich'l weact sickant ndy dtone?

grance:
banke:
in tacceivad i are-:
thou mokno:
lexvone that.

durkenos:
becround foret on tarke ua;
the tains ad athitedus age puthar ward fo; bikf harplemer;
your damasath, fean these i: wis the shincore;
the wislfste.

jullale:

his menormels oof tho beall,
an

'''
