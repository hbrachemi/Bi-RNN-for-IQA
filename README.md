### On the Use of Bi-RNN for Image Quality Assessment

## Usage
The source code is available in the notebook.
* Please either download the datasets or update the DataGenerator's parametters when creating a an instance of it.
* Please note that the Data Generator expects the IDs to be a list stored in a pickle file.

1. Install dependencies, import required libraries and download required datasests.
2. Create an instance of the data generator as follows:
 ```python 
 training_generator = DataGenerator(list_IDs_path='./IDs.pickle',overlapping=0,
                     db_path='./Koniq/512x384/',batch_size=1,dim=(224,224), n_channels=3,
                     n_output=1, shuffle=False, part='train',base='resnet')
 val_generator = DataGenerator(list_IDs_path='./IDs.pickle',overlapping=0,
                     db_path='./Koniq/512x384/',batch_size=1,dim=(224,224), n_channels=3,
                     n_output=1, shuffle=False, part='test',base='resnet')
```
3. Create an instance of the CNN network:  
```python 
 base_model =  Base_Model('resnet',weights='imagenet', include_top=False, input_shape=(224, 224, 3))     
```
Then build the feature extractor model.
4. Extract the features using the predict function:
```python
 X_train = model_cnn.predict_generator(generator=training_generator)
 X_test = model_cnn.predict_generator(generator=val_generator)
```
5. Define the RNN model
6. Load y and train the model on the train set.

