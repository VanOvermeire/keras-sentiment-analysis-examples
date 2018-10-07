from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from functions import data_preparer

EPOCHS = 20
INPUT_SHAPE = 10_000
TRAINING_LIMIT = 20_000
VALIDATION_LIMIT = 32_000

one_hot_results, encoded_labels, tokenizer = data_preparer.get_data_as_one_hot(INPUT_SHAPE)

training_data = one_hot_results[:TRAINING_LIMIT]
training_labels = encoded_labels[:TRAINING_LIMIT]

validation_data = one_hot_results[TRAINING_LIMIT:VALIDATION_LIMIT]
validation_labels = encoded_labels[TRAINING_LIMIT:VALIDATION_LIMIT]

evaluation_data = one_hot_results[VALIDATION_LIMIT:]
evaluation_labels = encoded_labels[VALIDATION_LIMIT:]

print('{} training, {} validation, {} evaluation data. '.format(len(training_data), len(validation_data), len(evaluation_data)))
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(INPUT_SHAPE,)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1, activation='sigmoid'))

print('Compiling and fitting model')
callbacks = [EarlyStopping(monitor='acc', patience=2),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
             ]

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(training_data,
                    training_labels,
                    epochs=EPOCHS,
                    batch_size=512,
                    validation_data=(validation_data, validation_labels),
                    callbacks=callbacks)

print('Testing')
results = model.evaluate(evaluation_data, evaluation_labels)
print(results)

print('Predicting some examples')
example_list = ['mooi en compacte pc werkt perfect. Fantastisch', 'heel slecht, meteen kapot', 'werkt helemaal niet', 'goedkoop, snel en mooi. aan te raden']
examples = tokenizer.texts_to_matrix(example_list, mode='binary')
results = model.predict_classes(examples)
count = 0

for result in results:
    print('Predicted {} for {}'.format(result, example_list[count]))
    count += 1

# Predicted [1] for mooi en compacte pc werkt perfect. Fantastisch
# Predicted [0] for heel slecht, meteen kapot
# Predicted [0] for werkt helemaal niet
# Predicted [1] for goedkoop, snel en mooi. aan te raden
