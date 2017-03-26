#**Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./visualization/histogram.png "Histogram"
[image2]: ./visualization/randomImage.png "Random Image"
[image3]: ./visualization/orig.png "Original Image Grid"
[image4]: ./visualization/proc.png "Preprocessed Image Grid"
[image5]: ./testImages/bike.png "Bicycle Crossing"
[image6]: ./testImages/childrenCrossing.png "Childredn Crossing"
[image7]: ./testImages/keepRight.png "Keep Right"
[image8]: ./testImages/noEntry.png "No Entry"
[image9]: ./testImages/yield.png "Yield"

---

Here is a link to my [project code](https://github.com/drzeinner/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.  

I used basic numpy functions to calculate the following summary of the data set

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

When exploring the data set the first thing I wanted to do was get an idea of what the distribution of classes looked like.
So I used a matplotlib histogram to plot the training and validation sets. 

This image is saved in visualizations/histogram.png.

![alt text][image1]

You can see that is quite a bit a disparity between certain classes. This made me wonder if the under represented classes were going to be harder to predict.

Next I just wanted to get a quick look a some random images in the data set, so I created some quick functionality to save out a random image.
Running this function a couple times gave me some ideas of the what to data looked like before I went onto the preprocessing steps.

This image is save in visualizations/randomImage.png.

![alt text][image2]

After I preprocessed my data set I came back to the data exploration portion of the code so that I could compare groups of original images with the preprocessed images.
I wrote a quick function to display a grid of random images and used it to display a grid of the the original data set and the preprocessed data set.

This image is save in visualizations/orig.png.

![alt text][image3]

This image is save in visualizations/proc.png.

![alt text][image4]

Saving out these images let me see if my preprocessing had bugs and if it was improving the contrast and overall visibility of the source images.

###Design and Test a Model Architecture

The code for this step is contained in the fourth code cell of the IPython notebook.

The first thing I tried was just running the network without any preprocessing. I wanted to set a baseline on how much the model could improve with preprocessing.
I also thought that 3 colors channels vs 1 could be helpful in classifying the signs. After running the model without any preprocessing I got about 83%-85% validation accuracy.
I decided I would next try to normalize the 3 colors channels between (-1, 1) to get a tighter distribution centered around 0. This immediately helped me get to 85% validation accuracy consistently, however the training accuracy was a lot higher.
This lead me to believe that the model was overfitting. So I grayscaled all of the images to see if reducing the amount of features would help fight overfitting. This ended up reducing my training accuracy but kept my validation about the same. This meant that it was slightly less overfitting.
After working with this for a while and not improving much I decided to go back and reinspect my data set to see if I could find anything to improve upon in the preprocessing pipeline. It turns out that there were quite a few very dark images where the signs were very hard to distinguish.
This made me want to apply a method to increase contrast, so I decided to use histogram equalization. This immediately got me just above 90% accuracy.

The code for splitting the data into training and validation sets is contained in the fourth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by combining the provided data set into one large dataset. Then I randomly shuffled the entire data set. After that I split the data set into train, validation and test sets.

The train set contains 60% of the total data set which was 31103 entries.
The validation set contains 20% of the total data set which was 10368 entries.
The test set contains the remaining entries which was 10368 entries.
My final training set had X number of images. My validation set and test set had Y and Z number of images.

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Dropout   	      	| 50%                            				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Dropout   	      	| 50%                            				|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 5x5x16 	|
| RELU					|												|
| Flatten               | outputs 400x1                                 |
| Fully connected		| 400x120      									|
| RELU					|												|
| Fully connected		| 120x84      									|
| RELU					|												|
| Dropout   	      	| 50%                            				|
| Fully connected		| 84x43      									|

The code for training the model is located in the sixth cell of the ipython notebook. 

To train my model I used an Adam Optimizer passing in the average cross entropy of the softmax probabilities of my network. 
The hyperparameters that I needed to fine tune included: # Epochs, Batch Size, Learning Rate, Dropout Percentage, L2 Regularization Scalar.
In order to determine the best combination of hyperparameters I set up my program so that I train it repeatedly passing in arrays of various hyperparameter values. 
Each time training the model I would save out learning curve plot that displayed the train/validation accuracy curves over the number of epochs.
After that I simply needed to choose the curve that had the lowest bias and variance.

The code for calculating the accuracy of the model is located in the sixth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 97.2%
* validation set accuracy of 96.7%
* test set accuracy of 96.2%

* I started by implementing the LeNet architecture that was described in our lab. This was actually pretty good from the start and got me about 80% validation accuracy.
* Next I decided I needed to plot the learning curves for the train/validation sets so I could see what I should focus on improving.
* The first thing I noticed was that both the train and validation accuracies were low, implying a high bias issue. This meant that I either needed more data or need to add more complexity to my model.
* I decided I would add more layers to the model. I decided to increase the number of filters outputted by my second convolutional layer to 32 and then reduce it back down to 16 using a 1x1 convolutional layer. This allowed me to calculate more features with my model and still keep the same expected output size.
* Next I noticed that my training accuracy was quite high yet my validation accuracy was low. This meant I had a high variance problem. I knew I either needed to reduce my model's complexity or introduce regularization. Since I had just increased the complexity I decided to go with regularization.
* I decided to implement both dropout and l2 regularization at the same time and tweak the numbers to find the best settings. After experimenting for a while I was able to get both my training and validation accuracies to above 96%.
 

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

* The first image might be difficult to classify because the sign is not centered and it includes some noise.
* The second image might be difficult to classify because the sign is not centered and the image is very large.
* The third image might be difficult to classify because the sign is warped and there is noise at the bottom.
* The fourth image might be difficult to classify because the sign is not centered and the image is overall bright.
* The fifth image might be difficult to classify because the sign is angled and there is noise in the image.

The code for making predictions on my final model is located in the seventh cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycle Crossing   	| Priority road   								| 
| Children Crossing 	| Turn left ahead								|
| Keep Right			| End of all speed and passing limits			|
| No Entry	      		| Wild animals crossing					 		|
| Yield			        | Yield      							        |


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This doesn't compare well the test accuracy of 96%.
I assume the model was having problems detecting signs that were farther away from the camera and thus appear smaller scale in the image. I think adding some data augmentation to introduce signs of various sizes could have helped to address this.
The model also seemed to have some trouble with image with low contrast. I used histogram equalization to attempt to address this, but maybe there are some better methods to increase image contrast that would have been better.

For the first image (the Bicycle Crossing) I got the following softmax probabilities

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| 12,Priority road   							| 
| .32     				| 17,No entry 									|
| .17					| 14,Stop										|
| .5	      			| 33,Turn right ahead					 		|
| .4				    | 15,No vehicles     							|

For the second image image (the Children Crossing) I got the following softmax probabilities

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 34,Turn left ahead   							| 
| .001     				| 35,Ahead only 								|
| .0005					| 38,Keep right									|
| .0004	      			| 30,Beware of ice/snow					 		|
| .0003				    | 20,Dangerous curve to the right      			|

For the third image image (the Keep Right sign) I got the following softmax probabilities

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .25         			| 32,End of all speed and passing limits   		| 
| .17     				| 18,General caution 							|
| .13					| 11,Right-of-way at the next intersection		|
| .12	      			| 28,Children crossing					 		|
| .08				    | 12,Priority road      						|

For the fourth image image (the No Entry sign) I got the following softmax probabilities

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .29         			| 31,Wild animals crossing   					| 
| .24     				| 20,Dangerous curve to the right 				|
| .06					| 30,Beware of ice/snow							|
| .05	      			| 23,Slippery road					 		    |
| .04				    | 10,No passing vehicles over 3.5 metric tons   |

For the fifth image image (the Yield sign) I got the following softmax probabilities

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 13,Yield   									| 
| .006     				| 12,Priority road 								|
| .004					| 14,Stop										|
| .00005	      		| 1,Speed limit (30km/h)					    |
| .000003				| 4,Speed limit (70km/h)      				    |