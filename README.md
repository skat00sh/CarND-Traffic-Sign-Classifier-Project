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

Training Accuracy(With Validation set created out of training data) = 96.9
Test Accuracy = 90.0
Validation Accuracy = 90/8

### Test a Model on New Images

Four Different type of signboards chosen from internet. Two of the images shown below are taken a from a recent paper that says how neural networks can be fooled.

![alt text][image8]
![alt text][image9] 
To make sure that

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (70km/h)       		| Speed limit (30km/h) 									|
| Yield     			| Yield 										|
| No entry				| No entry										|
| No entry				| No entry										|
| Stop      			| Stop     		    							|
| Speed limit (70km/h)	| Speed limit (30km/h)							|
| Keep Right			| Priority road									|

The model was able to correctly predict 5 other 7 traffic signs, which gives an accuracy of 71%.

Based on the comparison with the accuracy of the testing sample (0.958) and the lower number of images for Speed limit 70 in contrast with other speed limits images I think the bad prediction of the max speed sign was due the small quantity of examples for this kind of images on the data sample. Adding variations of the images by inverting, rotating or augmenting the them might have increased the accuracy.  

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For my images 1, 2, 3 and 5 my model was 100% sure that the results were correct. For image number 3 the probability was 79.41% and still got it right.
For images 6 and 7 the probability was 93.64% and 92.08% but it got them wrong. For image number 6 the second option was the correct one but for image number 7 the correct solution wasn't listed in the top 5 softmax.

image1.png:
No entry: 100.00%
Speed limit (120km/h): 0.00%
Traffic signals: 0.00%
Stop: 0.00%
Children crossing: 0.00%

image2.png:
Yield: 100.00%
Priority road: 0.00%
No passing for vehicles over 3.5 metric tons: 0.00%
Speed limit (50km/h): 0.00%
Speed limit (20km/h): 0.00%

image3.png:
No entry: 79.41%
Stop: 9.97%
Traffic signals: 4.87%
Children crossing: 4.53%
No vehicles: 1.23%

image4.jpg:
No entry: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

image5.png:
Stop: 100.00%
Bicycles crossing: 0.00%
Road work: 0.00%
Bumpy road: 0.00%
Children crossing: 0.00%

image6.png:
Speed limit (30km/h): 93.64%
Yield: 6.25%
Speed limit (50km/h): 0.11%
Speed limit (20km/h): 0.00%
Dangerous curve to the right: 0.00%

image7.png:
Priority road: 92.08%
Keep right: 7.54%
Speed limit (30km/h): 0.21%
Speed limit (20km/h): 0.18%
Roundabout mandatory: 0.00%

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The challenge here was to name the convolutional step correctly so you can reference it from the get_tensor_by_name() function.q

After that it was easy to show one or more steps on the network.

Here is my results at the end of the first and second convolution using my web images:

![alt text][image5]

## Conclusion:
### After watching lessons 10 and 11 I found some ways I could improve this project.

1 - I would reduce the number of epochs to prevent it from going up and down on the prediction accuracy.

2 - I think the bad prediction of the max speed sign was due the small quantity of examples for this kind of images on the data sample. Adding variations of the images by inverting, rotating or augmenting the them might have increased the accuracy.
