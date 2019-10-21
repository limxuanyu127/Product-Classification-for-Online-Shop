#FASHION MODEL

#text model
json_file = open('fashion_text_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
fashion_text_model = model_from_json(loaded_model_json)
fashion_text_model.load_weights("fashion_text_model.h5")
print("Loaded model from disk")
last_layer = Dense(512, activation='relu')(fashion_text_model.layers[-3].output)
fashion_text_model = Model(fashion_text_model.input, last_layer)

for layer in fashion_text_model.layers[:-1]:
    layer.trainable = False
fashion_text_model.summary()
#image model
last_layer = vgg16_model.get_layer('fc2').output
output = keras.layers.Dense(512, activation="relu",name='output')(last_layer) 
fashion_image_model = keras.models.Model(inputs=vgg16_model.inputs, outputs=output)
for layer in fashion_image_model.layers[:-1]:
    layer.trainable = False
fashion_image_model.summary()
#print(type(fashion_text_model.output))

#merged model    
merged_model = concatenate([fashion_text_model.output,fashion_image_model.output])
merged_model = Dense(128,activation="relu",name="fc_after_concat")(merged_model)
merged_model = Dense(num_fashion, activation="softmax")(merged_model)
fashion_model = Model([fashion_text_model.input, fashion_image_model.input],merged_model)
fashion_model.compile(loss='categorical_crossentropy',optimizer = 'adadelta', metrics=['accuracy'])
fashion_model.summary()
