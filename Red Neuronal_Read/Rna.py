from __future__ import absolute_import, fivision, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()

logger.setlevel(logging.ERROR)

# Obtenemos el set de datos y metadatos del set, utilizando la libreria de tensorflow metadata que ya los tiene incluidos
#La libreria se llama 'mnist'

dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)

#Obtenemos los 60.000 datos de entrenamiento con dataset como train_dataset y los 10.000 para evaluacion en test_dataset

train_dataset, test_dataset = dataset['train'], dataset['test']

#Definimos etiquetas de texto simples para cada uno delas posibles respuestas de nuestra red

class_name = [
    'cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis',
    'siete', 'ocho', 'nueve'
]

#Obtenemos la cantidad de ejemplos en variables para utilizarlos despues (los 60.000 y 10.000)

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#Definimos funcion de normalizacion
#Con esto vamos a hacer que el valor de rango de los pixeles sea de 0 a 1 y no de 0 a 255.
#Lo hacemos dividiendo el numero por 255

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# Llamamos a la funcion de normalizacion en cada dato de ambos set de datos

train_dataset = train_dataset.map(normalize)
train_dataset = test_dataset.map(normalize)

#Definimos la estructura de la red
#Indicamos prinerio la capa de entrada con 784 neuronas, indicando que llegara en una forma cuadrada de 28por 28
#Luego agregamos dos capas ocultas y densas con 64 neuronas cada una 
#Luego la capa de salida

model = tf.keras.Sequential([
    tf.keras.layers.Fllaten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) #Para clasificacion
])
#Por el moemnto utilizaremos las funcion de activacion relu para las capas ocultas y la funcion de activacion softmax para la capa de salida

#Compilamos el modelo, configurando la funcion de costo a utilizar y el optimizador. Tambien por ultimo la presicion que queremos obtener con 'accuracy'

model.compile(
    optimizer = 'adam',
    loss = 'sparce_categorical_crossentropy',
    metric = ['accuracy']
)

#Preparamos los lotes de datos que utilizamos. Aprendizaje por lotes de 32 datos cada lote
#Especificamos un tamaño de lote de 32

BATCHSIZE = 32

#Ordenamos los datos de entrenamiento de forma aleatoria

train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
#LLenamos y especificamos el tamaño del lote
test_dataset = test_dataset.batch(BATCHSIZE)


#Entrenamiento (Realizar el entrenamiento)
#Especificamos los datos del entrenamiento
#Cuantas epocas o vueltas completas a los datos del set se usaran en el entrenamiento

model.fit(
    train_dataset, epochs= 5,
    steps_per_epoch = math.ceil(num_train_examples/BATCHSIZE)
)

#Evaluamos nuestro modelo ya entrenado, para determinar que tan entrenada quedo la red

test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples/32)
)

#Imprimimos el resultado

print(f'Resultado en las pruebas: {test_accuracy}')


#Vemos resultado en forma grafica

for test_images, test_labels in test_dataset.take(1):
	test_images = test_images.numpy()
	test_labels = test_labels.numpy()
	predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_labels, images):
	predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img[...,0], cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("Prediccion: {}".format(class_names[predicted_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#888888")
	plt.ylim([0,1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

numrows=5
numcols=3
numimages = numrows*numcols

plt.figure(figsize=(2*2*numcols, 2*numrows))
for i in range(numimages):
	plt.subplot(numrows, 2*numcols, 2*i+1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(numrows, 2*numcols, 2*i+2)
	plot_value_array(i, predictions, test_labels)

plt.show()
