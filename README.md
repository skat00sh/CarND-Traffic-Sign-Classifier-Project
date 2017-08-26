## Project: Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
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

[image1]: ./vis/dataset.png "Visualization 1"
[image2]: ./vis/Traffic_sign_frequency.png "Visualization 2"
[image3]: ./new_test_images/i1_0.png "Traffic Sign 1"
[image4]: ./new_test_images/i1_17.png "Traffic Sign 2"
[image5]: ./new_test_images/i3_4.png "Traffic Sign 2"
[image6]: ./new_test_images/i4a_14.png "Traffic Sign 2"
[image7]: ./new_test_images/i4b_14.png "Traffic Sign 2"
[image8]: ./new_test_images/i4c_14.png "Traffic Sign 2"
[image9]: ./new_test_images/i4d_14.png "Traffic Sign 2"
## Rubric Points
### Please find below Validation of each [rubric points](https://review.udacity.com/#!/rubrics/481/view) below.
---

### Data Set Summary & Exploration

#### 1. Basic summary of dataset 	

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410

* Image data shape = (32, 32, 3)

* Number of classes = 43

#### 2. Exploratory visualization of the dataset.

All Sign boards displayed.

![alt text][image1]

Bar chart showing Frequency of each traffic sign in training set.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Pre-processing of data
Shuffling of dataset using the sklearn.utils library.
X_train, y_train = shuffle(X_train, y_train)

Normalizing of images using Adaptive Histogram equalization

#### 2. CNN Architecture used
Modified LeNet for 43 labels


| Layer         		|     Description	        					                | Activation Function |
|:---------------------:|:---------------------------------------------:|:-------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6 	|	RELU					|											
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6   |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x16  | RELU					|												
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x16    |
| Flatten				| output = 400									|
| Fully connected	1	| input = 400, output = 120       	            | RELU					|												
| Fully connected	2	| input = 120, output = 84       	            | RELU					|												
| Fully connected	3	| input = 84, output = 10       	            |


epochs = 10  
Batch size = 128 
learning rate = 0.001.

Training Accuracy(With Validation set created out of training data) = 96 (average)
Test Accuracy = 91 (average)
Validation Accuracy = 91 (average)

### Test a Model on New Images

Four Different type of signboards chosen from internet. Two of the images shown below are taken a from a recent paper that says how neural networks can be fooled.

![alt text][image8]
![alt text][image9] 


Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (20km/h)       		| Speed limit (30km/h) 									|
| Stop    			  | Stop 										|
| Stop (Noisy)				| Speed limit (30km/h)			|
| Stop (Noisy)				| Priority road		  			|
| Stop(With Background)    			  | Stop 										|
| No entry				| No entry										|
| Speed limit (70km/h)	| Speed limit (30km/h)							|

The model was able to correctly predict 4 other 7 traffic signs, which gives an accuracy of 57%.
 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

All the Softmax probabilities have been plotted alongside images.
The top5 softmax probabilities for new images is shown below:

i1_0.png: Correct Answer : Speed limit (20km/h)
Speed limit (30km/h): 99.14%
Speed limit (20km/h): 0.64%
Speed limit (80km/h): 0.19%
Speed limit (70km/h): 0.02%
Speed limit (50km/h): 0.00%

i4a_14.png: Correct Answer : Stop
Stop: 99.84%
No vehicles: 0.10%
Speed limit (80km/h): 0.03%
Speed limit (60km/h): 0.03%
No entry: 0.00%

i1_17.png: Correct Answer : No entry
No entry: 100.00%
Stop: 0.00%
Speed limit (20km/h): 0.00%
Speed limit (70km/h): 0.00%
Bicycles crossing: 0.00%

i4c_14.png: Correct Answer : Stop
Speed limit (60km/h): 36.26%
Speed limit (80km/h): 25.92%
Roundabout mandatory: 25.19%
Speed limit (30km/h): 3.91%
Go straight or right: 3.74%

i4d_14.png: Correct Answer : Stop
Speed limit (30km/h): 94.24%
Speed limit (20km/h): 3.65%
Speed limit (50km/h): 1.33%
Stop: 0.56%
Speed limit (60km/h): 0.06%

i4b_14.png: Correct Answer : Stop
Stop: 99.92%
No entry: 0.07%
Speed limit (30km/h): 0.01%
Speed limit (20km/h): 0.00%
Speed limit (80km/h): 0.00%


i3_4.png: Correct Answer : Speed limit (70km/h)
Speed limit (70km/h): 80.18%
Speed limit (30km/h): 19.78%
Speed limit (80km/h): 0.02%
Speed limit (20km/h): 0.02%
Speed limit (50km/h): 0.01%

