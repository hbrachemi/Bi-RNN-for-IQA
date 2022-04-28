# On the Use of Bi-RNN for Image Quality Assessment

## Contents
1. [Abstract](#Abstract)
2. [Performance Benchmark](#Performance-Benchmark)

    2.1. [Scores comparison on authentic distortions' databases](#Scores-comparison-on-authentic-distortions'-databases)
   
    2.2. [Scores comparison on synthethic datasets](#Scores-comparison-on-synthethic-datasets)
3. [Model Zoo](#Model-Zoo) 
4. [Usage](#Usage)
5. [Contact](#Contact)



## Abstract
The deployment of Deep Neural Networks (DNNs) based on Convolutional Neural Networks (CNNs) pipelines as feature extractors has led to an impressive rise of performance on different computer vision tasks. However many challenges are encountered while dealing with DNNs in the No Reference Image Quality Assessment (NR-IQA) context of which in particular the non uniform distribution of a global quality across the different areas of the the assessed images. We propose a Bi-directional Recurrent Neural Network (RNN) approach that aims to overcome this issue.


  ![](https://github.com/hbrachemi/Bi-RNN-for-IQA/blob/main/BiRCNN.drawio.png)

## Performance Benchmark

### Scores comparison on authentic distortions' datasets

#### Live in the wild dataset

|Metric| SROCC ↑| PLCC ↑|KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|BRISQUE| 0.5710 | 0.5954 | 0.4034
|NIQE| 0.3879 | 0.4274 | 0.2637
|VGG16| 0.8058 | 0.7996 | 0.6189
|Bi-RCNN(VGG16)| 0.8223 | 0.8568 | 0.6368
|Resnet50| 0.8186 | 0.8361| 0.6341
|Bi-RCNN(Resnet)| 0.8229 | 0.8514| 0.6357

#### Koniq dataset

|Metric| SROCC ↑| PLCC ↑| KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|BRISQUE| 0.5710 | 0.5954 | 0.4743
|NIQE| 0.3879 | 0.4274 | 0.0536
|VGG16| 0.8058 | 0.7996 |0.5906
|Bi-RCNN(VGG16)| 0.8223 | 0.8568 | 0.6160
|Resnet50| 0.8186 | 0.8361| 0.6468
|Bi-RCNN(Resnet)| 0.8229 | 0.8514| 0.6669

### Scores comparison on synthethic datasets

#### Live dataset

|Metric| SROCC ↑| PLCC ↑| KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|PSNR| 0.8907 | 0.9221 | 0.7236
|SSIM| 0.8895 | 0.9130 | 0.7239
|GMSD| 0.9447 | 0.9439 | 0.8086
|FSIM| 0.9517 | 0.9444 | 0.8260
|VIF|  0.9017 | 0.9235 | 0.7326
|BRISQUE| 0.9382 | 0.9475 | 0.7878
|NIQE| 0.6668 | 0.6440 | 0.4743
|VGG16| 0.9605 | 0.9646 | 0.8305
|Bi-RCNN(VGG16)| 0.9826 | 0.9783 | 0.8945
|Resnet50| 0.9652 | 0.9706 | 0.8418
|Bi-RCNN(Resnet)| 0.984 | 0.9861 | 0.9036

#### Kadid10k dataset

|Metric| SROCC ↑| PLCC ↑| KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|PSNR| 0.6910 | 0.6969 | 0.5016
|SSIM| 0.6796 | 0.6698 | 0.4969
|GMSD| 0.8467 | 0.8461 | 0.6619
|FSIM| 0.8353 | 0.8317 | 0.6453
|VIF|  0.6345 | 0.6360 | 0.4681
|BRISQUE| 0.6378 | 0.6585 | 0.4695
|NIQE| 0.3380 | 0.3994 | 0.2279
|VGG16| 0.9412 | 0.9482 | 0.7884
|Bi-RCNN(VGG16)| 0.9581 | 0.9595 | 0.8262
|Resnet50| 0.9385 | 0.9394 | 0.7852
|Bi-RCNN(Resnet)| 0.9638 | 0.9657 | 0.8369

#### CSIQ dataset

|Metric| SROCC ↑| PLCC ↑| KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|PSNR| 0.7776 | 0.8054 | 0.5736
|SSIM| 0.7647 | 0.7536 | 0.5680
|GMSD| 0.9580 | 0.9526 | 0.8164
|FSIM| 0.9226 | 0.9086 | 0.7612
|VIF|  0.6901 | 0.7609 | 0.5073
|BRISQUE| 0.8219 | 0.8588 | 0.6457
|NIQE| 0.6459 | 0.6600 | 0.4605
|VGG16| 0.9753 | 0.9818 | 0.8684
|Bi-RCNN(VGG16)| 0.9781 | 0.9791 | 0.8758
|Resnet50| 0.9809 | 0.9852 | 0.8877
|Bi-RCNN(Resnet)| 0.9829 | 0.9849 | 0.8909

#### TID2013 dataset

|Metric| SROCC ↑| PLCC ↑| KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|PSNR| 0.6912 | 0.6832 | 0.4964
|SSIM| 0.6198 | 0.6243 | 0.4393
|GMSD| 0.8108 | 0.8535 | 0.6375
|FSIM| 0.8043 | 0.8529 | 0.6275
|VIF|  0.5844 | 0.6959 | 0.4305
|BRISQUE| 0.7510 | 0.7754 | 0.5684
|NIQE| 0.2526 | 0.2842 | 0.1675
|VGG16| 0.9421 | 0.9492 | 0.7956
|Bi-RCNN(VGG16)| 0.9526 | 0.9547 | 0.815
|Resnet50| 0.9357 | 0.9441 | 0.7843
|Bi-RCNN(Resnet)| 0.9596 | 0.9648 | 0.8307

#### Live Multi Distortions dataset

|Metric| SROCC ↑| PLCC ↑| KRCC ↑|
|------|:-------------:|:---------------:|:---------------:|
|PSNR| 0.5300 | 0.5682 | 0.3842
|SSIM| 0.4563 | 0.4856 | 0.3072
|GMSD| 0.8500 | 0.8666 | 0.6591
|FSIM| 0.8658 | 0.8814 | 0.6786
|VIF|  0.6309 | 0.6479 | 0.4716
|BRISQUE| 0.8363 | 0.8474 | 0.6586
|NIQE| 0.4692 | 0.6170 | 0.3492
|VGG16| 0.9712 | 0.9695 | 0.8635
|Bi-RCNN(VGG16)| 0.9704 | 0.9664 | 0.8565
|Resnet50| 0.9788 | 0.9752 | 0.8815
|Bi-RCNN(Resnet)| 0.9734 | 0.9741 | 0.865

## Model Zoo
weights of the GRU RNN tested on five different folds are also available along with their [corresponding IDs splits](https://drive.google.com/drive/folders/1LeMyb1XokmZm16_aRflWQZ8fD-z4UXV9?usp=sharing) [here](https://drive.google.com/drive/folders/19dAWLv75Qz3wXWKwOpkbT4AYf22O0Yaa?usp=sharing).

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

## Contact 
Hanene F.Z Brachemi Meftah , `hbrachemi@inttic.dz`

Sid Ahmed Fezza , `sfezza@inttic.dz`
