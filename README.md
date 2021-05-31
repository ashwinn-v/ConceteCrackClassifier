## ConceteCrackClassifier

#### A good and well-maintained Infrastructure plays a crucial role in the economic and overall development of a country. Therefore it is vital to properly maintain these civil engineered structures, especially the concrete bridges and roads, as they are more prone to structural decay or collapse due to various defects in their infrastructure and other environmental factors. We have built a binary image classification model that can detect the cracks in concrete surfaces with greater accuracy by using methods like Image Augmentation and Transfer Learning.

### The Dataset

The dataset that we use for training our model consists of 40000 images, of which 20000 of them are the images of concretes surfaces from buildings and bridges having cracks in them, and the other 20000 are the ones having no cracks in them. Each image having RGB color channels are rescaled to 227x227 pixels. All the images were again converted to 150x150 pixels when fed into the model. We divide the available data into training and validation sets to train the model using the training set and evaluate them using the validation set. We split the dataset into training and validation in the 8:2 ratio.

<p align="center">
<img style="display: block; margin: auto;"
<img width="590" alt="Screenshot 2021-05-31 at 4 32 29 PM" src="https://user-images.githubusercontent.com/52974732/120183792-ccc96680-c22d-11eb-8770-579ee4713711.png">
 </p>
 
### CNN
Our first model consists of four 2D convolutional layers having the ‘relu’ activation function. Each of the convolutional layers is followed by pooling layers. These layers are then followed by a dense layer containing 512 nodes containing the ‘relu’ activation function. The last layer in our neural network consists of a dense layer having a single node with the sigmoid activation function. We compile our binary classifier model with the binary cross-entropy loss function. We use the RMS prop optimizer along with the accuracy as our metric.

### SVM
Here we trained our dataset on a linear classification SVM model. The kernel function used in the SVM model was the Radial Basis Function Kernel. The training time for the SVM model war very high compared to the other models due to the usage of complex kernel functions. So to speed up the training time we use the min-max scaler to scale our image vector to decrease the training time to some extent.

### Improved CNN

The previously trained CNN model reached the desired accuracy but the number of epochs taken to achieve that accuracy was high. So image augmentation techniques are done to the data set to achieve the desired result at a relatively less amount of training time. These augmentation techniques are done inside the memory and hence the size of the dataset
does not change. The images are augmented with a rotation angle of 40 degrees, The range of the height and width of the image is shifted by 0.2 range. Changes were also made to the shear range and zoom range. The horizontal flip augmentations was also set to true.

### VGG16
The transfer learning approach is utilized to improve the accuracy of the classifier model. This is done by freezing the already trained layers and adding them to our training layers. The state of the art model developed by A. Zisserman and K.Simonyan  achieves an accuracy of 92.7% on the ImageNet dataset. The pretrained model weights are used on the normal dataset as well as on the datasets altered using image augmentation techniques.

### EfficientNetB5
The EfficientNetB5  model was developed by scaling up the baseline network of EfficientNetB0 using an inverted bottleneck along with Depthwise Convolution operations. This neural network architecture proved to be much faster and smaller when compared to earlier network architectures line ConvNet and is expected to give better results compared to other models.

### InceptionV3
This network architecture is a convolutional neural network that is mainly used for object detection and image analysis. This architecture originated as a module for built for the GoogLeNet architecture. The Inception V3 consists of 48 neural layers and is trained over the ImageNet database which consists of more than 1 million images and has almost 1000 image classes. The InceptionV3 architecture used in our model is represented in Fig. 4.

### Xception

The neural network architecture developed by outperforms the InceptionV3 model mentioned earlier. The Xception model uses the same number of parameters as used by the InceptionV3 model but with more effective use of these model parameters resulting in an increase in model accuracy. This model is also trained over the ImageNet dataset and can classify images over a diverse range of classes. The Xception model architecture is represented in the below figure:

<p align="center">
<img style="display: block; margin: auto;"
<img width="704" alt="Screenshot 2021-05-31 at 4 37 42 PM" src="https://user-images.githubusercontent.com/52974732/120184476-a821be80-c22e-11eb-9d6a-3f4e9a0ccfe8.png">
<p


