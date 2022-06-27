**LAYERS:-**

**Layer 1: Reshape layer** 

The first layer added is the reshape layer that took an input of (4096, 1) and reshaped it into (64, 64, 1).

**Layer 2 - Layer 5: Convolution layer**

(i)The function of convolution layer is to find out distinctive features from the given input matrix which is an image of sign which has been already normalized.<br />
(ii)The convolution layer uses a filter matrix which is a combination of zeros and ones.<br />
(iii)The filter matrix slides over the input matrix and keeps the count of elements which matched with the elements of the filter matrix. <br />
(iv)The matrix so created is called feature map. This convolution layer takes input matrix of shape of (64, 64,1 ). <br />
(v)This layer uses a filter of size (3, 3) and a stride size of 1. It creates 32 feature maps as output using 32 different filters. It uses ReLU as the activation function.<br />


**Layer 6: Max pooling layer**

(i) Max pooling is done by applying a max filter to (usually) non-overlapping sub regions of the initial representation.<br />
(ii) When the images are too large, it would need to reduce the number of trainable parameters. <br />
(iii) Pooling is done for the sole purpose of reducing the spatial size of the image. Here it have used a pool size of (2,2) with stride size 2.<br />

**Layer 7: Flatten layer**

The function of the flatten layer is convert all elements of the feature maps matrices to individual neurons which will serve as input to the next layer.<br />

**Layer 8: Dense layer**

(i) Dense or the input layer which accepts the output from the flatten layer as input. <br />
(ii) The value held by a neuron is called activation of that neuron. Every unit of input (neuron) has activation corresponding to intensity of pixel. <br />
(iii) The output of this layer is determined using the activation function which is ReLU in this case. <br />
(iv) The function of activation function is to activate the neurons of the dense layer. This layer has 128 units. <br />
(v) Every neuron has a bias associated with it. <br />
(vi) The neurons are activated based on the outcome from the activation function, if its activation is greater than 0.5 the neuron is activated else the neuron remains inactive.<br />

**Layer 9: Dropout layer**

(i) The function of the dropout layer is to remove some of the neurons or the unwanted features that can make the model bulky and increase the training time. <br />
(ii) It is also helping in avoiding over fitting. A dropout layer which removes 50 percent of neurons was used here.<br />

**Layer 10: Dense layer**

The last layer of the model is the dense layer which is also called the output layer. The last layer has 20 neurons corresponding to 20 different hand gestures.<br />
     
                      TABLE 1 - MODEL SUMMARY
                      
MODEL : " SEQUENTIAL_1 "

| LAYER TYPE        | OUTPUT SHAPE       | PARAMS  |
| :-------------:   |:-------------:     | :-----: |
| conv2d_4 (Conv2D)          | (None, 62, 62, 32)      | 320   |
| max_pooling2d_4 (MaxPooling2D)          | (None, 31, 31, 32)           |   0   |
| conv2d_5 (Conv2D)     | (None, 29, 29, 32)           |    9248   |
| max_pooling2d_5 (MaxPooling2D)     | (None, 14, 14, 32)           |    0   |
| conv2d_6 (Conv2D)     | (None, 12, 12, 32)           |    9248   |
| max_pooling2d_7 (MaxPooling2D)     | (None, 6, 6, 32)           |    0  |
| conv2d_7(Conv2D)     | ((None, 4, 4, 32)           |    9248   |
| max_pooling2d_7 (MaxPooling2D)     | (None, 2, 2, 32)           |    0   |
| flatten_1 (Flatten)     | (None, 128)           |    0   |
| dense_2 (Dense)     | (None, 128)           |    16512   |
| dropout (Dropout)     | (None, 128)           |    0   |
| dense_3 (Dense)     | (None, 20)           |    2580   |
| Total Trainable params : |                 |       47156|
-----------------------------------

                                       BLOCK DIAGRAM
![BLOCK DIAGRAM](https://user-images.githubusercontent.com/78750216/175886183-71054e65-0be6-40c3-95f5-90def8e9ca14.png)
                            
**TRAINING PROCESS :** <br/>
The training on a personal computer (RAM - 4 GB) took about 3 hours with 20 units in dense layer. The model trained with dataset which had been pre- processed and optimized for training beforehand. Later it has categorically distributed the train and test labels and flattened out train and test arrays for easy input into model. Then stuck with a model with 4 Convolution layers as it gave us the highest accuracy. Model accuracy increased with each epoch.

At first, it increased exponentially, and later had a minute and steady growth. At the end of training, the result obtained was 95.93 percent accuracy which is a little shy of the maximum accuracy obtained using the dataset. The evaluation of the model was done with the test data provided. This accuracy can be increased further in future research work by pre-processing the dataset even more and by adding new hyper parameters to the keras model. The lower the loss, the better the model (unless the model has over fitted to the training data). The calculation of loss on train and test data was done. In case of neural networks, the loss is usually negative log-likelihood and the residual sum of squares for classification and regression respectively. Then naturally, objective was to reduce the loss 20 functions value with respect to the model’s parameters. The loss with the model during training began at 2.24 with the first epoch and ended up at 0.21.

As the model chapter previously stated, classifier contained two Convolution and a single Max pooling layer with a dropout layer and a dense layer with 512 units. There was a tried tweaking this by adding a few more back to back convolution layers but this didn’t work well for the model and resulted in a lower accuracy.


                       

