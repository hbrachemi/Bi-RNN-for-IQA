### On the Use of Bi-RNN for Image Quality Assessment
## Abstract
The deployment of Deep Neural Networks (DNNs) based on Convolutional Neural Networks (CNNs) pipelines as feature extractors has led to an impressive rise of performance on different computer vision tasks. However many challenges are encountered while dealing with DNNs in the No Reference Image Quality Assessment (NR-IQA) context of which in particular the non uniform distribution of a global quality across the different areas of the the assessed images. We propose a Bi-directional Recurrent Neural Network (RNN) approach that aims to overcome this issue.


  ![](https://github.com/hbrachemi/Bi-RNN-for-IQA/blob/main/BiRCNN.drawio.png)

## Performance Benchmark
# Scores comparison on authentic distortions' databases
*Live in the wild dataset*

|Metric| SROCC ↑| PLCC ↑|KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|BRISQUE| 0.5710 | 0.5954 | 0.4034
|NIQE| 0.3879 | 0.4274 | 0.2637
|VGG16| 0.8058 | 0.7996 | 0.6189
|Bi-RCNN(VGG16)| 0.8223 | 0.8568 | 0.6368
|Resnet50| 0.8186 | 0.8361| 0.6341
|Bi-RCNN(Resnet)| 0.8229 | 0.8514| 0.6357

*Koniq dataset*

|Metric| SROCC ↑| PLCC ↑| KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|BRISQUE| 0.5710 | 0.5954 | 0.4743
|NIQE| 0.3879 | 0.4274 | 0.0536
|VGG16| 0.8058 | 0.7996 |0.5906
|Bi-RCNN(VGG16)| 0.8223 | 0.8568 | 0.6160
|Resnet50| 0.8186 | 0.8361| 0.6468
|Bi-RCNN(Resnet)| 0.8229 | 0.8514| 0.6669


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

