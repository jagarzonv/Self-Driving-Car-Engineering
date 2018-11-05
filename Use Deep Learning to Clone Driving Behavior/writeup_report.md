
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/netarchitecture.jpg "Model architecture"
[image2]: ./images/modelresume.jpg "nvidia model resume"
[image3]: ./images/train1.jpg "train image 1"
[image4]: ./images/train2.jpg "train image 2"
[image5]: ./images/trainres1.jpg "train response 1"
[image6]: ./images/loss1.png "train curve 1"
[image7]: ./images/trainres2.jpg "train response 2"
[image8]: ./images/loss2.png "train curve 2"
[image9]: ./images/testc1.jpg "train curve non inv "
[image10]: ./images/testc2.jpg "train curve inv"
[image11]: ./images/graph1.jpg "graph steering 1"
[image12]: ./images/graph2.jpg "graph steering 2"
[image13]: ./images/graph3.jpg "graph steering 3"

## Rubric Points


# Model Architecture and Training Strategy

## 1. An appropriate model arcthiecture has been employed

I implement NVIDIA architecture[1] shown in next figure (solution code, lines 85 to 105),  this demostrates better performance in simulator tests than other proved like  VGG'16'  for same data base provided in project repository named IMG_Udacity. In the top I added a fully connected layer with one output  and in the normalization block I include color normalization and input image crop (160x320  to 80x320 ) to reduce the model dimension parameters.


![alt text][image1]

[1] End to End Learning for Self-Driving Cars,Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba, 

## 2. Attempts to reduce overfitting in the model

Aditionally I include one dropout layer in the fully connected block  as a regularization method in order to reduce overfitting effects (line 98).  

I probe the model based on three data sets, first data set correspond to provides by Udacity in the project repository (8036 samples), the second data set is obtained with Play station 3 game controller obtaining 5141 samples. The final data set is obtained with PS3 controler in opposite direction  (3929 samples). Te model parameters was tuned so that the results were similar in each data set (lines of code: 181 to 211). In each data collection process I follow the lesson recommendations moving the vehicle to obtain the greatest variability of the data.


## 3. Model parameter tuning

The adam optimizer was used with learning rate of 1e-3, other values of this parameter was probed from 1e-4 to 1e-2 but in the first case an overfitting is present , mainly by including steering with zero values. this was evident in the second and third curve where the car failed and was off the road. In the second case the behavior did not fit with the training data. 

Batch size and Epoch parameters were also adjusted through numerous tests. The batch takes values from 8 to 64 in each test with five epochs , low values causes underfitting  and for  high values  overfitting is present , finally a batch size of 32 presents a better response. In Epoch parameter I adjust a value of 5 , for higtest values the squared error loss  decay decay was minimum.



## 4. Appropriate training data

The first attempt for create the training data was using the keyboard  of PC. Five different people were asked to perform three laps of the first track (10000 samples aprox) but the results were not promising. So I opted to use a  Play Station 3 game controller with higher resolution to generate two sets of data each with 10 to 12 laps. In both cases combined positions in the center of the road with zigzag movements.  Each data set was taken in different driving directions (5141 and 3929 samples in each one). I use xboxdrv --detach-kernel-driver  solution proposed in post found in [2]

[2] : http://askubuntu.com/questions/409761/how-do-i-use-a-ps3-sixaxis-controller-with-ubuntu-to-control-games 

![alt text][image3]
![alt text][image4]

The above images show samples of training data. the simulator was configured in fastest mode with 640x480 lower resolution

## 5. Solution Design Approach

First I work with modified version of VGG architecture from the previous lesson, however I had problems with my GPU resources  "Resource exhausted error" , Then I quickly migrated to NVIDIA architecture learned in this lesson, This model shows a great initial behavior without major changes.  

In a second phase I use Udacity  data set  in experimental process where I note a overfitting problems betwen training and validation  sets , also not good results were obtained when testing with the second track. At this point I decide probe the generalization capacity of NVIDIA algorithm by only training with data from track 1 and testing the system with the two tracks. In this phase two layers was included in the network input, one for image normalization and other for image crop based in lesson of this course.

The third phase includes data augmentation. In order to obtain more data I apply a keras generator (lines 37 to 80)  to expand the data set with the inclusion of inverted images and directions and using left and right camera images.

In a fourth phase I execute aditional recording data  process (with track1 only) obtaining two aditional data sets. New data set was used together with  Udacity set for comparison purposes. Three data sets was independently used in training - validation and test experiments. At this point the system parameters were adjusted to obtain similar performance in each data set, also a dropout layer was included. However, there were still problems with second track where the system does not respond properly to narrow corners


In a final phase I apply the tansfer learning lessons for fine tuning the model (lines 155 to 177) by probe the combination betwen data sets assuming a large data set - similar data case *. I found that the  use of Udacity and non inverted PS3 data sets in a transfer learning approach shows adequate behavior in both tracks working with a speed value=3 in a drive.py file. After some tests the objective was achieved by first training with the data set of udacity for a total of 5 epochs with a strong adjustment in the left and right camera angles followed by a fine adjustment with PS3 data for 2 epochs with slight adjustment of left and right camera angles.

The next images shows the training - validation resume for initial and fine tuning process.

![alt text][image5]
![alt text][image6]

For fine tuning :

![alt text][image7]
![alt text][image8]


* "If the new data set is large and similar to the original training data:
remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
1)randomly initialize the weights in the new fully connected layer
2)initialize the rest of the weights using the pre-trained weights
3)re-train the entire neural network"

## 6. Final Model Architecture

The next image shows the model architecture of NVIDIA  covnet used in this project , I replicate the architecture from the lesons in this module. Only three changes were made:  include one  layer for image normalization at the input of the network (lambda_1) ,  next include one layer for image cropping (cropping2d_1) and  finally include a dropout layer in the set of fully connected layers (drop1).

![alt text][image2]

The activation functions is defined by a Rectified Linear Units (RELUs), A set of 5 convolutional layers is kept unchanged as well as  the 4 fully connected layers.


## 7. Creation of the Training Set & Training Process

I create two aditional data sets,  one in the default drive direction and other in opposite way, in each data set approximately 12 laps were taken  including laps with center lane driving, left and right driving and zig zag driving. in both data sets the simulator was configured at 640 x 480 resolution and fastest graphics performance, a play station 3 game controller was used to smooth data drive 

Data Set 1 = PS3     5141 samples
Data Set 2 = PS3 inv 3929 samples 


Next images show a  PS3 and PS3 inv images in same curve:

![alt text][image9]
![alt text][image10]

In the next image the graphs shows the three data set used in this project , the first graph shows the steering angles for each sample  for the udacity repository data , this serves as reference point for data recording.
![alt text][image11]
The next two graphs shows the data collected with simulator in training mode where zig zag movement is performed for default and opposite way respectively:
![alt text][image12]
![alt text][image13]

Then the total steering angles in data sets are :

for udacity data =  8036   

for ps3 data     =  5141  

for ps3 inv data =  3929 

This data sets are split in training data and validation data  in 80 /20 proportion, thus :

for udacity data :    (train) =  6428    , (validation) = 1607

for ps3 data:         (train) =  4112    , (validation) = 1028

for ps3 inv data:     (train) =  2351    , (validation) =  785 

The data sets are augmented through a keras generator. The generator shuffle the samples and returns the left , right and center images from set and additionally flip each one. Thus the total samples in training are:

for udacity data augmented:    (train) =  38568    , (validation) = 9642

for ps3 data augmented:        (train) =  24672    , (validation) = 6168

for ps3 inv data augmented:    (train) =  18858    , (validation) = 4710

The trainig process uses adam optimizer and mse error function, the model and weights are saved in *.h5 files  and plot the training and validation loss for each epoch (lines 109 to 151). Finally the test is made with the simulator in autonomous mode (640x480, fastest graphics) , the model was successfully tested in both tracks, for first track I runs two laps  , one with throttle = 0.2 and other with throttle = 0.3  and for second track with throttle = 0.3. three videos are included to show this results.


```python

```
