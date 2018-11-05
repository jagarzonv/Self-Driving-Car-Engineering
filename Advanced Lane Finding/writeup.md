
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Writeup


### Camera Calibration

Through Opencv findChessboardCorners() function, I uses as reference the code called example.ipynb  found in example folder , in this code first define the number of inner intersection points in the calibration pattern , in this case  6 vertical poins and 9 horizontal points. findChessboardCorners() function  placed order: row by row, left to right in every row the corner point fond [1]. The info of x, y image coordinates of  points (6*9=54 points per image) found in each image are stored in imgpoints array and objpoints contains the coordinates of the points respect to calibration pattern. The code for this step (the total code for this project is in the IPython notebook named Code.ipynb) is located at lines 27 to 67 
The next image shows the corner detection results.


#### 1. References:

[1]  http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners


[imagecorners]: ./output_images/opencv_corner_detection.png "Some corner detection images"
![alt text][imagecorners]

With corner detection info above I use cv2.calibrateCamera function based on [2,3] to obtain the intrinsic and extrinsic parameters of the camera as well as the distortion parameters of radial and tangential distortion (line 59 of code):

ret =returns the root mean square (RMS) re-projection error
mtx =intrinsic parameters
dist=distortion coefficients
rvecs,tvecs = extrinsic parameters

[2]Zhang. A Flexible New Technique for Camera Calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000
[3]J.Y.Bouguet. MATLAB calibration tool. http://www.vision.caltech.edu/bouguetj/calib_doc/

The results of camera calibration are saved in calibration_results.p file (line 67)

### Pipeline (single images)


#### 1. Example of a distortion-corrected image.

I apply the cv2.undistort to obtain a dst image with distortion correction, first load the calibration_results.p file  and extract the intrinsic camera paremeters from matrix mtx and distortion coefficients from dist field (lines 71 to 73) , next feed parameters to cv2.undistort (line 74) . The  next images shows two examples of distorded input and undistorded result (The single images pipeline is found in lines 1 to 28 of penultimate cell).

[imageundistorded]: ./output_images/undistorded_images.png "Some corner undistorded images"
![alt text][imageundistorded]


#### 2. Thresholding of input  image. 
The threshold function is found at line 143 of code , inside I apply color space transformation (split RGB, convert to gray , HSL (Hue ,Lightness,Saturation), and  HSV (Hue, Saturation, Value)) where I found the best results with HSV and select the V channel (line 152). After channel select I apply thresholding functions based on gradient (abs_sobel_thresh2 , line 81) in x and y directions,    gradient magnitude (mag_thresh2,line 100)  and a threshold color channel (line 165), each one with adjusted parameters based on test images results. The thresholded result is an sum of individual functions in order to obtain the majority of pixels that form the lane  lines (line 169).  Also a morphological open filter was applyed for noice reduction (lines 171 , 172). An image example is shown bellow:

[imagethresholded1]: ./output_images/thresholded_image1.jpg "Some thresholded images 1"
![alt text][imagethresholded1]

[imagethresholded2]: ./output_images/thresholded_image2.jpg "Some thresholded images 2"
![alt text][imagethresholded2]

[imagethresholded3]: ./output_images/thresholded_image3.jpg "Some thresholded images 3"
![alt text][imagethresholded3]





#### 3. Perspective transform:

The perspective transform is  found in lines 202 to 236. Here I define four polygon vertices (line 208)  as a src points based on straight_lines1.jpg and straight_lines2.jpg found in test images folder. The destination points is selected based on size image plus offset (lines 216 to 222). 

Source and destination points:

Source : 
 [  580.   450.]
 [  700.   450.]
 [ 1152.   719.]
 [  156.   719.]

Destination:
 
 [  200.     0.]
 [ 1080.     0.]
 [ 1080.   720.]
 [  200.   720.]

The next image shows the polygon mask  defined by source points and defined order: 


[imagemask]: ./output_images/sourcepoints.jpg "mask from source points"
![alt text][imagemask]

The source and destination points are the parameters of the     M = cv2.getPerspectiveTransform(src, dst) function (line 224) that calculates the perspective transform matrix. Also I obtain the inverse perspective transform matrix through cv2.getPerspectiveTransform(dst, src)  function  and use (line 226) and use cv2.warpPerspective function (line 228) to run the interpolation (flags=cv2.INTER_LINEAR) betwen selected src and dst perspectives. The next images shows some perspective transform results ("birds-eye view") after normalization (lines 229 to 231) and binarization  (line 232) steps. 

[imagewrap1]: ./output_images/perspective_transform1.jpg "Some transformed images 1"
![alt text][imagewrap1]

[imagewrap2]: ./output_images/perspective_transform2.jpg "Some transformed images 2"
![alt text][imagewrap2]

[imagewrap3]: ./output_images/perspective_transform4.jpg "Some transformed images 3"
![alt text][imagewrap3]

#### 4. Lane-line polynomial fitting

The polynomial fit is processed in detect_lane_lines function (lines 516 to 804), the histogram based initialization  (lines 526 to 535) and sliding windows approach (lines 538 to 591) in the module lesson is used to find points correspond to lane lines on the road, the search parameters are 9 windows , minimum number of pixels per window = 50 , window margin =100 . 


[imagewindows1]: ./output_images/searchwindows1.jpg "windows search 1"
![alt text][imagewindows1]

[imagewindows2]: ./output_images/searchwindows2.jpg "windows search 2"
![alt text][imagewindows2]

[imagewindows3]: ./output_images/searchwindows3.jpg "windows search 3"
![alt text][imagewindows3]


The lane line points are stored in lefty, leftx , righty, rightx arrays and use np.polyfit function to fit points to a second order polynomial which has the form:

f(y) = Ay^2 + By + C  

A = curvature of the lane line
B = direction that the line is pointing
C = position respect image y axis

The np.polyfit function returns  A, B, and C coefficients (lines 594 to 595). The curvature coefficients also was calculated in world space through  (lines 597 and 598) with meters per pixel in x and y dimension constants. In addition, the values of the coefficients are averaged to soften the measurements, thus creating a more stable approximation of the results (lines 640 to 643)


The next images shows the polynomial line fitting (in yellow) for some images.

[imagefitt1]: ./output_images/polyfitt1.jpg "fitt 1"
![alt text][imagefitt1]

[imagefitt2]: ./output_images/polyfitt3.jpg "fitt 2"
![alt text][imagefitt2]

[imagefitt3]: ./output_images/polyfitt4.jpg "fitt 3"
![alt text][imagefitt3]

#### 5. Radius of curvature and vehicle offset measures.

I create a  measures function (line 240) for calculate the radius and offset properties , here the curvature in image space (lines 258 and 259 ) and world space (lines 266 and 267 ) is obtained through :

curv =((1 + (2*A*y + B)**2)**1.5) / |2*A|

The offset of the vehicle respect to lane center  is measured based on distance betwen the center of image and the x coordinates of the lane lines (lines 278 and 279). the difference of lane line distances relatives to image center is obtained in line 281 and is transformed to world space through the defined  conversions in x and y from pixels space to meters:

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

#### 6. Lane Area Visualization.

The visualization process is implemented in function project_lines() ( lines 290 to 346) , here I define four colors for lane area for evaluation pourposes of sanity check function (lines 349 to 512)  and LookAheadFilter flag (line 814). 

The sanity check verifies that the increase between coefficients of the polynomial function is bounded  between consecutive frames (lines 356 to 382), also verifies if Left and Right curvature coefficients are similar (lines 392 to 405), check  if lane lines are separated by approximately the right distance horizontally (lines 408 to 414) and  check that the lane lines  are roughly parallel (lines 418 to 438) . Sanity check  is pass if all conditions are  met.

The  LookAheadFilter flag indicates that the line search process was run again or around the previous detection

color blue    =  sanity check "pass"  and LookAheadFilter "new search"
color red     =  sanity check "fails" and LookAheadFilter "new search"
color green   =  sanity check "pass"  and LookAheadFilter "around previous"
color magenta =  sanity check "fails" and LookAheadFilter "around previous"

Also render text information on image output that indicates the curvature and offset measures (lines 320 to 344)

The next image shows the result output image in each case 

[imageout1]: ./output_images/imageout1.jpg "image out 1"
![alt text][imageout1]

[imageout2]: ./output_images/imageout2.jpg "image out 2"
![alt text][imageout2]

[imageout3]: ./output_images/imageout3.jpg "image out 3"
![alt text][imageout3]

[imageout4]: ./output_images/imageout4.jpg "image out 4"
![alt text][imageout4]

### Pipeline (video)

The final resul is called outputvideo.mp4  and is generated in lines 2 to 5 of  last cell  of Code.ipynb

Here's a [link to my video result](outputvideo.mp4)


### Discussion

In this project I follow step by step the course  lessons  as well as the sample code and tips and tricks provided. The principal problems occurred during the threshold step since a fine adjustment of the parameters is required to obtain acceptable results for the project video. this is a basic approach but the results  change drastically for the challenge_video and  harder_challenge_video, this may imply making a new adjustment for each case which is not scalable.

A similar fine-tuning was made at the source and target points for the perspective transformation which leads to problems when the camera pose is different from the example.

An important aspect of the implementation was the Sanity Check and Look-Ahead Filter algorithms which eliminated much of the noise generated when measurements is doing for each frame without any control

In accordance with the above, it can be concluded that for the result video, acceptable results were obtained both in the detection of the lane lines and the approximation to the curvature and distance of the vehicle but it is not a general solution that can work well for changes in:

Color and intensity of the lane lines
Changes of lighting on the road
Road color
Road patches
Road marking
Highway without marking !!!
Noise in the image due to poor Highway  condition
Vehicles oclude lane lines

This is evident in the roads of my country where a much more robust solution is required. It is very interesting research on machine vision algorithms to detect lines in uncontrolled conditions like a contrast or illumination normalizations, I hope to address more robust algorithms in upcoming opportunities  


```python

```
