# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images_output/1_TrainSetDistribution.jpg
[image2]: ./test_images_output/1_ValidationSetDistribution.jpg
[image3]: ./test_images_output/1_TestSetDistribution.jpg
[image4]: ./test_images_output/1_OriginalImage_ChannelVisulization.jpg
[image5]: ./test_images_output/1_TrainSetDistribution.jpg
[image6]: ./test_images_output/1_TrainSetDistribution.jpg
[image7]: ./test_images_output/1_TrainSetDistribution.jpg
[image8]: ./test_images_output/1_TrainSetDistribution.jpg
[image9]: ./test_images_output/3_GTSRBTrafficSign.jpg
[image10]: ./test_images_output/3_Top5BarPlot.jpg
[image11]: ./test_images_output/3_Top5Possibility.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of unique classes/labels = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distrbuted:

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to keep the image as BGR, because BGR could identify shape in low light condition, shown below:
![alt text][image4]

As a last step, I normalized the image data because I need to take out the variance of the different lighting condition. In addition, I also center the image because CNN performs better with symmetric data. Also the pixel data is divided by the 3*sigma of each color channel, because CNN performs better with data range from -1 to +1.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer 	        |    Description      		|
|:-----------------:|:-------------------------:|
| **Input**  		| 32x32x3 RGB image   							|
| **Layer 1**      		|							|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| dropout               | 90% droprate         	                        |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| **Layer 2**      		|							|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x6 	|
| RELU					|												|
| dropout               | 90% droprate         	                        |
| Max pooling	      	| 2x2 stride,  outputs 5x5x6  			    	|
| **Layer 3**      		|							|
| Flat             	    | outputs 400  									|
| Fully connected		| outputs 120        							|
| RELU					|												|
| **Layer 4**      		|							|
| Fully connected		| outputs 84        							|
| RELU					|												|
| **Layer 5**      		|							|
| Fully connected		| outputs 43        							|
| Softmax				|           									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, the **optimizer** I used is AdamOptimizer. I have implemented degrading **learning rate**, and it would decrease to 90% if 4 consecutive epoches accuracy increasing < 1%. I also increase the **epoches** to 30. Tabulate below:
* EPOCHS = 30
* BATCH_SIZE = 100
* LEARN RATE = 0.001 (initial)
* DROPOUT = 0.9
* OPTIMIZER = AdamOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 94.2%
* test set accuracy of 85.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * Initial architecture is listed below:
        * 2x (conv2d + RELU + maxpool)
        * 2x (full conv + RELU + dropout)
        * 1x (full conv + RELU)
* What were some problems with the initial architecture?
    * The initial architecture cannot only reach training accuracy to 86% and validation accuracy to 77.3%. Percentage difference is at 8.7%, which is really high --> overfitting
    * Both dropout & RELU at full conv remove the negative or zero weights and biases, shifting the parameters to positive. Therefore, the gradient is no long correct.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * 2nd iteration architecture is listed below:
        * 2x (conv2d + RELU + maxpool + dropout)
        * 2x (full conv + RELU)
        * 1x (full conv + RELU)
    * Final iteration architecture is listed below:
        * 2x (conv2d + RELU + maxpool + dropout)
        * 2x (full conv + RELU)
        * 1x (full conv)
    * The reason for not dropping parameters at full conv or output layer is that full conv doesn't have neurons

* Which parameters were tuned? How were they adjusted and why?
    * **Epoches**: increasing epoches number help a lot. **Reason**: more iteration.
    * **Learn Rate**: Learn rate has been trailed from 0.0006 to 0.001. Decreasing learning rate doesn't help any as initial expected. It decreases both accuracy. **Reason**: not sure why dropping learn rate not helping.
    * **Dropout Rate**: dropout rate has been tested from 0.3 to 0.9. It drastically increases the accuracy when changed from 0.3 to 0.9. **Reason**: dropout rate decrease overfitting --> increase validation accuracy.
    * **Batch size**: help with validation accuracy. **Reason**: decreasing batch size help decrease overfitting --> increase validation accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Dropout rate help with overfitting. Therefore, the training model would be more general comparing to no dropout rate, and eventually it would fit the validation set better.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9]

The Label= 29(Bicycles crossing) might be difficult to classify because it only has 240 sample in training set.

| Label |     Description	 | Num. of Sample in Training Set		|
|:-------------:|:-------------------:|:----------------------:|
| 1   		| Speed limit (30km/h)  			|		1980|
| 9     			| No passing				|		1320|
| 23				| Slippery road				|		450	|
| 29      		| Bicycles crossing				|	 	240	|
| 31		| Wild animals crossing    			|		690	|



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |Prediction	      	|
|:---------------------:|:-------------------:|
| Speed limit (30km/h)  | Speed limit (30km/h) 		|
| No passing   			| No passing				|
| Slippery road		|Slippery road		|
| Bicycles crossing		| Bicycles crossing		|
| Wild animals crossing	| Wild animals crossing	|



The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 85.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Probability, Top 1			        |Prediction	      	|
|:---------------------:|:-------------------:|
| 99.81%  | Speed limit (30km/h) 		|
| 100.00%   			| No passing				|
| 99.51%		|Slippery road		|
| 99.99%	| Bicycles crossing		|
| 99.96%	| Wild animals crossing	|

**1st Image**

The model is pretty sure that this is a Speed limit (30km/h) (probability of 99.81%), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were:
* 0.999957	Speed limit (30km/h)
* 4.112967e-05	Speed limit (80km/h)
* 1.877737e-06	Speed limit (50km/h)
* 4.697173e-09	Wild animals crossing
* 5.359590e-11	Speed limit (60km/h)

**3rd Image**

The model is pretty sure that this is a Slippery road (probability of 98.37%), and the image does contain a Slippery road. The top five soft max probabilities were:
* 0.983754	Slippery road
* 1.059639e-02	Dangerous curve to the right
* 2.311018e-03	Speed limit (120km/h)
* 1.107668e-03	Traffic signals
* 1.086815e-03	No passing

**5th Image**

The model is pretty sure that this is a Wild animals crossing (probability of 99.63%), and the image does contain a Wild animals crossing. The top five soft max probabilities were:
* 0.996375	Wild animals crossing
* 3.625018e-03	Double curve
* 1.346337e-09	Bicycles crossing
* 5.986092e-10	Dangerous curve to the left
* 3.167339e-10	Speed limit (50km/h)

For **2nd Image**, **4th Image**, the model has really high accuracy of prediction. See plot below:
![alt text][image11]
![alt text][image10]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
