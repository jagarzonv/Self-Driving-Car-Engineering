# Imports
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt
import pickle

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# Early in the code before pipeline
polygon_points_old = None


#ploting function 
def plot_results(original_img, result_img):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(result_img, cmap='gray')
    ax2.set_title('Result Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
def plot_results2(original_img,title1, result_img,title2):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original_img)
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(result_img, cmap='gray')
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    
# Camera Calibration 
def calibrate_camera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)#define object to save corners coordinates
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')# pointer to images 

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# convert color images to gray space 

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)#obtain  u, v coordinates of corners 

        # If found, add object points, image points
        if ret == True:  #The function returns a non zero value if detect all corners , otherwise returns zero 
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows() 

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) #obtain camera parameters
    #ret,mtx,dist


    # Save the camera calibration result 
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "calibration_results.p", "wb" ) )
    
#function that takes an distorded image and return an undistorded one
def undistord_frame(image,cal_results_path): 
    calbration_values = pickle.load(open(cal_results_path, "rb"))
    dist=calbration_values['dist']
    mtx=calbration_values['mtx']
    undistorded = cv2.undistort(image, mtx, dist, None, mtx) # undistorded images
    return undistorded

# Functions that takes an image threshold (x,y) gradient,gradient magnitude (min / max values) and 
# gradient orientation for a given sobel kernel size and threshold values
#threshold (x,y)gradient :

def abs_sobel_thresh2(img, orient='x',sobel_kernel=3,mag_thresh=(210, 255)):
    #1. Take the derivative in x or y given orient = 'x' or 'y'
    #2. Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=sobel_kernel))

    #3. Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    #4. Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    #5. Return the result
    return binary_output

#gradient magnitude (min / max values) :
def mag_thresh2(img, sobel_kernel=3, mag_thresh=(0, 255)):
    #1.  Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #2. Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    #3. Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    #5. Return the binary image
    return binary_output

# gradient orientation:
def dir_threshold2(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    #1. Calculate the x and y gradients separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #2. Take the absolute value of the gradient direction, 
    #   Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    if not sobelx.all():
        print("threshold x is empty")
        binary_output =  np.zeros_like(img)
        return binary_output
    elif not sobely.all():
        print("threshold y is empty")
        binary_output =  np.zeros_like(img)
        return binary_output
    else:
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        #3. Rescale back to 8 bit integer
        #   scaled_absgraddir = np.uint8(255*absgraddir/np.max(absgraddir))
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        #4. Return the binary image
        return binary_output


#threshold main function
def threshold_frame(image): 
    #Space color transformatios
    R, G, B = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    h_, l_, s_ = cv2.split(hls)# or hls[:,:,2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
    h, s, v = cv2.split(hsv)
    #Select the work channel and denoise
    channel=cv2.GaussianBlur(v, (5,5), 0)
    #Select the kernel size , Choose a larger odd number to smooth gradient measurements
    ksize = 7 
    #Apply each of the thresholding functions with adjusted parameters
    gradx = abs_sobel_thresh2(channel, orient='x', sobel_kernel=ksize, mag_thresh=(0,10))
    grady = abs_sobel_thresh2(channel, orient='y', sobel_kernel=ksize, mag_thresh=(90,200))
    mag_binary = mag_thresh2(channel, sobel_kernel=ksize, mag_thresh=(90, 250))
    #dir_binary = dir_threshold2(channel, sobel_kernel=ksize, thresh=(0.7,np.pi/2))
    #thresholding_stack= np.dstack( (np.zeros_like(s_binary),s_binary,grady,gradx,mag_binary,dir_binary))
    #combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    #Threshold color channel
    s_thresh=(220,255)
    s_binary = np.zeros_like(gradx)
    s_binary[(channel >= s_thresh[0]) & (channel <= s_thresh[1])] = 1

    #thresholded_result=(255-gradx)|grady|s_binary|mag_binary
    #################################################################################
    HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # For yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))
    # For white
    sensitivity_1 = 68
    white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))
    sensitivity_2 = 60
    HSL = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
    white_3 = cv2.inRange(image, (200,200,200), (255,255,255))
    thresholded_result = (255-gradx)|grady|mag_binary |s_binary | yellow | white | white_2 | white_3  
    ################################################################################# 
    #filter
    fker=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    filter_result = cv2.morphologyEx(thresholded_result, cv2.MORPH_OPEN, fker,iterations = 1)
    #Return the binary image
    return filter_result    
    

#apply a polygon mask    
def region_of_interest(img, vertices):
    #Applies an image mask.
    #Only keeps the region of the image defined by the polygon
    #formed from `vertices`. The rest of the image is set to black.

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#Perspective transform
def perspective_transform_frame(image): 
    # Grab the image shape
    xcorrection=30
    ycorrection=5
    imshape = (image.shape[0], image.shape[1])
    #define the polygon vertex
    vertices = np.array([[(186-xcorrection,imshape[0]-1),
                          ((imshape[1]/2)-30-xcorrection, (imshape[0]/2)+95-ycorrection), 
                          ((imshape[1]/2)+30+xcorrection, (imshape[0]/2)+95-ycorrection),
                          (imshape[1]-158+xcorrection,imshape[0]-1)]],
                           dtype=np.int32)    
    #maskimg=region_of_interest(image, vertices)
    #plot_results(image,image*maskimg)
    
    offsetx=200
    offsety=0
    src = np.float32([vertices[0,1],vertices[0,2],vertices[0,3],vertices[0,0]])
    dst = np.float32([[offsetx, offsety], 
                     [imshape[1]-offsetx, offsety], 
                     [imshape[1]-offsetx, imshape[0]-offsety], 
                     [offsetx, imshape[0]-offsety]])
    #print(src,dst)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Inverse perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped_result = cv2.warpPerspective(image,M, (imshape[1],imshape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT,borderValue=(0, 0, 0))#,flags=cv2.INTER_LINEAR
    scale_factor = np.max(warped_result)/255 
    warped_scale = (warped_result/scale_factor).astype(np.uint8)
    warped_mask = np.zeros_like(warped_scale)
    warped_mask[(warped_scale == 255)] = 1
    #Return the binary image
    result=(warped_mask ,Minv)
    #plot_results(image,warped_mask) 
    return result

#Curvature and offset measures
def measures(out_img,checkflag):

    
    lbx=L_Lane_Line.bestx
    rbx=R_Lane_Line.bestx
    
    
    l_fit=L_Lane_Line.best_fit
    r_fit=R_Lane_Line.best_fit

    l_fitw=L_Lane_Line.best_fitw
    r_fitw=R_Lane_Line.best_fitw
        
    
    #Curvature in image space: 
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(out_img)-10
    left_curverad = ((1 + (2*l_fit[0]*y_eval + l_fit[1])**2)**1.5) / np.absolute(2*l_fit[0])
    right_curverad =((1 + (2*r_fit[0]*y_eval + r_fit[1])**2)**1.5) / np.absolute(2*r_fit[0])
    #print(left_curverad, right_curverad)
    
    #curvature in world space

  
    # Calculate the new radii of curvature
    left_curverad  = ((1 + (2*l_fitw[0]*y_eval*L_Lane_Line.Ypixmeterprop + l_fitw[1])**2)**1.5) / np.absolute(2*l_fitw[0])
    right_curverad = ((1 + (2*r_fitw[0]*y_eval*R_Lane_Line.Ypixmeterprop + r_fitw[1])**2)**1.5) / np.absolute(2*r_fitw[0])
    # Now our radius of curvature is in meters
    #print( curvature[0], 'm',  curvature[1], 'm')

    L_Lane_Line.radius_of_curvature=left_curverad
    R_Lane_Line.radius_of_curvature=right_curverad
    #avg curvature      
    avgcurv=( L_Lane_Line.radius_of_curvature+R_Lane_Line.radius_of_curvature)/2
    
    #offset
    
    R_Lane_Line.line_base_pos=abs(rbx[out_img.shape[0]-5]-(out_img.shape[1]/2))*R_Lane_Line.Xpixmeterprop
    L_Lane_Line.line_base_pos=abs(lbx[out_img.shape[0]-5]-(out_img.shape[1]/2))*L_Lane_Line.Xpixmeterprop

    offset= L_Lane_Line.line_base_pos - R_Lane_Line.line_base_pos               
       
    measures=(avgcurv, offset)
    
    
    if (checkflag==True):
        L_Lane_Line.currentmeasures=measures 

#visualization
def project_lines(image,undist,warped,Minv,ploty,left_fitx,right_fitx,curv,offset,checkflag,checklookahead):
    
    global polygon_points_old
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    
    if  (polygon_points_old == None):
         polygon_points_old = np.int_([pts])
        
    a = polygon_points_old
    b = np.int_([pts])
    
    #print(a[0])
    #print(a.shape,b.shape)
    ret = cv2.matchShapes(a[0],b[0],1,0.0)
    
    if (ret < 0.095):
        # Use the new polygon points to write the next frame due to similarites of last sucessfully written polygon area
        polygon_points_old = np.int_([pts])
        
            # Draw the lane onto the warped blank image
    
        if (checklookahead==True):
    
            if(checkflag==True):#Sanity check flag  
                cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0)) #blue color if all search proces was required and sanity check pass
            else:
                cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0)) #red color if sanity check fails
        else:
            if(checkflag==True):#Sanity check flag  
                cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0)) #green color if search is fast and sanity check pass
            else:
                cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0)) #magenta color if fast search but sanity chack fails 

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        #plt.imshow(result)  
        #show text results in video
    

    else:
        # Use the old polygon points to write the next frame due to irregularities
        # Then write the out the old polygon points
        # This will help only use your good detections
        if (checklookahead==True):
    
            if(checkflag==True):#Sanity check flag  
                cv2.fillPoly(color_warp, polygon_points_old, (0,255,0)) #blue color if all search proces was required and sanity check pass
            else:
                cv2.fillPoly(color_warp, polygon_points_old, (0,255,0)) #red color if sanity check fails
        else:
            if(checkflag==True):#Sanity check flag  
                cv2.fillPoly(color_warp, polygon_points_old, (0,255,0)) #green color if search is fast and sanity check pass
            else:
                cv2.fillPoly(color_warp, polygon_points_old, (0,255,0)) #magenta color if fast search but sanity chack fails 

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        #plt.imshow(result)  
        #show text results in video        
    
    
    
    #show text results in video
    cv2.putText(result,"Curvature radius: ", (10,50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 2, 
                        color = (0,255,0),
                        thickness = 3)
    cv2.putText(result,str(round(curv, 2)), (600,50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 2, 
                        color = (0,255,0),
                        thickness = 3)
    cv2.putText(result,"m", (900,50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 2, 
                        color = (0,255,0),
                        thickness = 3)
    
    cv2.putText(result,"Car offset : ", (10,100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 2, 
                        color = (0,255,0),
                        thickness = 3)
    cv2.putText(result,str(round(offset, 2)), (400,100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 2, 
                        color = (0,255,0),
                        thickness = 3)
    cv2.putText(result,"m", (600,100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 2, 
                        color = (0,255,0),
                        thickness = 3)
    
    return result

#perform sanity checks 
def SanityCheck(img,checkflag):

    # 0 :Check if fit polynomial coeffs are consistent from last to current
    lcoefsflag=False
    #print(L_Lane_Line.diffs[0],L_Lane_Line.diffs[1],L_Lane_Line.diffs[2])
    
    # the next sanity check reference values was  obtained from project test images  folder           
    if (L_Lane_Line.diffs[0]>=0 and L_Lane_Line.diffs[0]<10e-5):
        if (L_Lane_Line.diffs[1]>=0 and L_Lane_Line.diffs[1]<50e-3):
            if (L_Lane_Line.diffs[2]>=0 and L_Lane_Line.diffs[2]<20):
                lcoefsflag=True
            else:
                lcoefsflag=False
        else:
             lcoefsflag=False
        
    else:
         lcoefsflag=False 
                
    rcoefsflag=False
    
    #print(R_Lane_Line.diffs[0],R_Lane_Line.diffs[1],R_Lane_Line.diffs[2])
                
    if (R_Lane_Line.diffs[0]>=0 and R_Lane_Line.diffs[0]<10e-5):
        if (R_Lane_Line.diffs[1]>=0 and R_Lane_Line.diffs[1]<50e-3):
            if (R_Lane_Line.diffs[2]>=0 and R_Lane_Line.diffs[2]<20):
                rcoefsflag=True
            else:
                rcoefsflag=False
        else:
             rcoefsflag=False
        
    else:
         rcoefsflag=False     
    
    
    
    
    
    if (checkflag==0):
                       
        
        # 1:Checking that they have similar curvature
        curvflag=False
        recentcurv=abs(L_Lane_Line.current_fit-R_Lane_Line.current_fit) #polynomial coefficients for the most recent fit
        
        if (recentcurv[0]>=0 and recentcurv[0]<10e-04):
            if (recentcurv[1]>=0 and recentcurv[1]<5e-01):
                if (recentcurv[2]>=0 and recentcurv[2]<10e+02):
                    curvflag=True
                else:
                    curvflag=False
            else:
                curvflag=False
        
        else:
            curvflag=False
        # 2: Checking that they are separated by approximately the right distance horizontally
        
        horizdistflag=False
        
        horiz_dist=abs(L_Lane_Line.allx[img.shape[0]-5]*L_Lane_Line.Xpixmeterprop-R_Lane_Line.allx[img.shape[0]-5]*R_Lane_Line.Xpixmeterprop)
        if (horiz_dist>=3.8 and horiz_dist<4.8):
                horizdistflag=True
        else:
                horizdistflag=False
            
        
        # 3: Checking that they are roughly parallel
        distflag=False
        Ldstbottom=L_Lane_Line.allx[img.shape[0]-5]
        Ldistmed=L_Lane_Line.allx[img.shape[0]/2]
        Ldisttop=L_Lane_Line.allx[5]
        Rdstbottom=R_Lane_Line.allx[img.shape[0]-5]
        Rdistmed=R_Lane_Line.allx[img.shape[0]/2]
        Rdisttop=R_Lane_Line.allx[5]
        
        dists=(abs( Ldstbottom- Rdstbottom) , abs( Ldistmed- Rdistmed),abs( Ldisttop- Rdisttop) )

        if (abs(dists[0]-dists[2])>=0 and abs(dists[0]-dists[2])<150):
            if (abs(dists[0]-dists[1])>=0 and abs(dists[0]-dists[1])<70):
                if (abs(dists[1]-dists[2])>=0 and abs(dists[1]-dists[2])<100):
                    distflag=True
                else:
                    distflag=False
            else:
                distflag=False
        
        else:
            distflag=False
            
        #varx=abs(np.std(L_Line.allx,axis=0)  -  np.std(R_Line.allx,axis=0))

        #print(recentcurv,horiz_dist,dists)
        
        
        if (curvflag==True&horizdistflag==True&distflag==True&lcoefsflag==True&rcoefsflag==True):#
                    return True
            
        else:
                    return False
            
    else:            
                
        
        # 1:Checking that they have similar curvature
        curvflag=False
        recentcurv=abs(L_Lane_Line.best_fit-R_Lane_Line.best_fit) #polynomial coefficients for the most recent fit
        
        if (recentcurv[0]>=0 and recentcurv[0]<10e-04):
            if (recentcurv[1]>=0 and recentcurv[1]<5e-01):
                if (recentcurv[2]>=0 and recentcurv[2]<10e+02):
                    curvflag=True
                else:
                    curvflag=False
            else:
                curvflag=False
        
        else:
            curvflag=False
        # 2: Checking that they are separated by approximately the right distance horizontally
        
        horizdistflag=False
        
        horiz_dist=abs(L_Lane_Line.bestx[img.shape[0]-5]*L_Lane_Line.Xpixmeterprop-R_Lane_Line.bestx[img.shape[0]-5]*R_Lane_Line.Xpixmeterprop)
        if (horiz_dist>=3.8 and horiz_dist<4.8):
                horizdistflag=True
        else:
                horizdistflag=False
            
        
        # 3: Checking that they are roughly parallel
        distflag=False
        Ldstbottom=L_Lane_Line.bestx[img.shape[0]-5]
        Ldistmed=L_Lane_Line.bestx[img.shape[0]/2]
        Ldisttop=L_Lane_Line.bestx[5]
        Rdstbottom=R_Lane_Line.bestx[img.shape[0]-5]
        Rdistmed=R_Lane_Line.bestx[img.shape[0]/2]
        Rdisttop=R_Lane_Line.bestx[5]
        
        dists=(abs( Ldstbottom- Rdstbottom) , abs( Ldistmed- Rdistmed),abs( Ldisttop- Rdisttop) )

        if (abs(dists[0]-dists[2])>=0 and abs(dists[0]-dists[2])<150):
            if (abs(dists[0]-dists[1])>=0 and abs(dists[0]-dists[1])<70):
                if (abs(dists[1]-dists[2])>=0 and abs(dists[1]-dists[2])<100):
                    distflag=True
                else:
                    distflag=False
            else:
                distflag=False
        
        else:
            distflag=False
            
        #varx=abs(np.std(L_Line.allx,axis=0)  -  np.std(R_Line.allx,axis=0))

        #print(recentcurv,horiz_dist,dists)
        
        
        if (curvflag==True&horizdistflag==True&distflag==True&lcoefsflag==True&rcoefsflag==True):#
                    return True
            
        else:
                    return False


# main processing function 
def detect_lane_lines(warped):
    
    smoothing=2
    
    if (L_Lane_Line.LookAheadFilter==False|R_Lane_Line.LookAheadFilter==False):
        
        
        
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
        #plt.plot(histogram)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped, warped, warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        
        
        
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = [] 

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 
        # Fit new polynomials to x,y in world space
        left_fitw =  np.polyfit( lefty*L_Lane_Line.Ypixmeterprop,  leftx*L_Lane_Line.Xpixmeterprop, 2)
        right_fitw = np.polyfit(righty*R_Lane_Line.Ypixmeterprop, rightx*R_Lane_Line.Xpixmeterprop, 2)
        
        L_Lane_Line.current_fit=left_fit
        R_Lane_Line.current_fit=right_fit
        L_Lane_Line.diffs = abs(L_Lane_Line.current_fit- L_Lane_Line.last_fit)
        R_Lane_Line.diffs = abs(R_Lane_Line.current_fit- R_Lane_Line.last_fit)        
        L_Lane_Line.last_fit=L_Lane_Line.current_fit
        R_Lane_Line.last_fit=R_Lane_Line.current_fit
    
        #Visualization

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
        
        #visualize
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #plt.imshow(out_img)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)

        
        L_Lane_Line.allx= left_fitx 
        R_Lane_Line.allx=right_fitx       
        L_Lane_Line.ally= ploty
        R_Lane_Line.ally= ploty             
        # polynomial coeffs of the last n fits of the line
        L_Lane_Line.recent_coeffs.append(left_fit)
        R_Lane_Line.recent_coeffs.append(right_fit)       
        L_Lane_Line.recent_coeffsw.append(left_fitw)
        R_Lane_Line.recent_coeffsw.append(right_fitw)
        # x values of the last n fits of the line
        L_Lane_Line.recent_xfitted.append(left_fitx)
        R_Lane_Line.recent_xfitted.append(right_fitx)           
        L_Lane_Line.counter= L_Lane_Line.counter+1
        R_Lane_Line.counter= R_Lane_Line.counter+1
        
        
        
        CheckFlag=SanityCheck(out_img,0) #sanity check test

        
        if (L_Lane_Line.counter>=smoothing and R_Lane_Line.counter>=smoothing ): 
            #average x values of the fitted line over the last n iterations      
            L_Lane_Line.bestx =np.average(L_Lane_Line.recent_xfitted, axis=0)         
            R_Lane_Line.bestx =np.average(R_Lane_Line.recent_xfitted, axis=0)  
            #polynomial coefficients averaged over the last n iterations
            L_Lane_Line.best_fit =np.average(L_Lane_Line.recent_coeffs, axis=0)         
            R_Lane_Line.best_fit =np.average(R_Lane_Line.recent_coeffs, axis=0)
            L_Lane_Line.best_fitw =np.average(L_Lane_Line.recent_coeffsw, axis=0)         
            R_Lane_Line.best_fitw =np.average(R_Lane_Line.recent_coeffsw, axis=0)
            
            
            L_Lane_Line.recent_coeffs=[]
            R_Lane_Line.recent_coeffs=[]
            L_Lane_Line.recent_coeffsw=[]
            R_Lane_Line.recent_coeffsw=[]
            # x values of the last n fits of the line
            L_Lane_Line.recent_xfitted=[]
            R_Lane_Line.recent_xfitted=[]           

            
            L_Lane_Line.counter=0
            R_Lane_Line.counter=0 
            Check_Flag=SanityCheck(out_img,1)
            measures(warped,Check_Flag)       
       
        
        CheckLookAhead=True
        
        if (CheckFlag==True):
             L_Lane_Line.LookAheadFilter=True
             R_Lane_Line.LookAheadFilter=True
             L_Lane_Line.fitcounter=L_Lane_Line.fitcounter+1
             R_Lane_Line.fitcounter=R_Lane_Line.fitcounter+1 
        else:
             L_Lane_Line.LookAheadFilter=False
             R_Lane_Line.LookAheadFilter=False 

        
        ret=(out_img,ploty,left_fitx ,right_fitx, L_Lane_Line.currentmeasures[0], L_Lane_Line.currentmeasures[1],CheckFlag,CheckLookAhead)
        return ret 
        
    elif(L_Lane_Line.LookAheadFilter==True & R_Lane_Line.LookAheadFilter==True):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_fit=L_Lane_Line.current_fit
        right_fit=R_Lane_Line.current_fit
        
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Fit new polynomials to x,y in world space
        left_fitw =  np.polyfit( lefty*L_Lane_Line.Ypixmeterprop,  leftx*L_Lane_Line.Xpixmeterprop, 2)
        right_fitw = np.polyfit(righty*R_Lane_Line.Ypixmeterprop, rightx*R_Lane_Line.Xpixmeterprop, 2)
        
        L_Lane_Line.current_fit=left_fit
        R_Lane_Line.current_fit=right_fit
        L_Lane_Line.diffs = abs(L_Lane_Line.current_fit- L_Lane_Line.last_fit)
        R_Lane_Line.diffs = abs(R_Lane_Line.current_fit- R_Lane_Line.last_fit)        
        L_Lane_Line.last_fit=L_Lane_Line.current_fit
        R_Lane_Line.last_fit=R_Lane_Line.current_fit
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        
        L_Lane_Line.allx= left_fitx 
        R_Lane_Line.allx=right_fitx       
        L_Lane_Line.ally= ploty
        R_Lane_Line.ally= ploty             
        # polynomial coeffs of the last n fits of the line
        L_Lane_Line.recent_coeffs.append(left_fit)
        R_Lane_Line.recent_coeffs.append(right_fit)
        L_Lane_Line.recent_coeffsw.append(left_fitw)
        R_Lane_Line.recent_coeffsw.append(right_fitw)
        # x values of the last n fits of the line
        L_Lane_Line.recent_xfitted.append(left_fitx)
        R_Lane_Line.recent_xfitted.append(right_fitx)           

        L_Lane_Line.counter= L_Lane_Line.counter+1
        R_Lane_Line.counter= R_Lane_Line.counter+1
        
        #visualize
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped, warped, warped))*255
        #window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        #left_line_pts = np.hstack((left_line_window1, left_line_window2))
        #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        #right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        

        
        CheckFlag=SanityCheck(out_img,0)# sanity check test
        
        if (L_Lane_Line.counter>=smoothing and R_Lane_Line.counter>=smoothing ): 
            #average x values of the fitted line over the last n iterations      
            L_Lane_Line.bestx =np.average(L_Lane_Line.recent_xfitted, axis=0)         
            R_Lane_Line.bestx =np.average(R_Lane_Line.recent_xfitted, axis=0)  
            #polynomial coefficients averaged over the last n iterations
            L_Lane_Line.best_fit =np.average(L_Lane_Line.recent_coeffs, axis=0)         
            R_Lane_Line.best_fit =np.average(R_Lane_Line.recent_coeffs, axis=0)
            L_Lane_Line.best_fitw =np.average(L_Lane_Line.recent_coeffsw, axis=0)         
            R_Lane_Line.best_fitw =np.average(R_Lane_Line.recent_coeffsw, axis=0)           

            
            L_Lane_Line.recent_coeffs=[]
            R_Lane_Line.recent_coeffs=[]
            L_Lane_Line.recent_coeffsw=[]
            R_Lane_Line.recent_coeffsw=[]
            # x values of the last n fits of the line
            L_Lane_Line.recent_xfitted=[]
            R_Lane_Line.recent_xfitted=[]           
            
            L_Lane_Line.counter=0
            R_Lane_Line.counter=0 
            Check_Flag=SanityCheck(out_img,1)
            measures(warped,Check_Flag)
        
              
      
        CheckLookAhead=False

        
        if (CheckFlag==True):
             L_Lane_Line.LookAheadFilter=True
             R_Lane_Line.LookAheadFilter=True
             L_Lane_Line.fitcounter=L_Lane_Line.fitcounter+1
             R_Lane_Line.fitcounter=R_Lane_Line.fitcounter+1            
        else:
             L_Lane_Line.LookAheadFilter=False
             R_Lane_Line.LookAheadFilter=False 


        ret=(out_img,ploty,left_fitx ,right_fitx, L_Lane_Line.currentmeasures[0], L_Lane_Line.currentmeasures[1],CheckFlag,CheckLookAhead)
        
        return ret
        
# class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #counter
        self.counter=0        
        #fit counter
        self.fitcounter=0
        #Look Ahead Filter
        self.LookAheadFilter=False
        # Define conversions in x and y from pixels space to meters
        # pixel-meter proportion
        self.Xpixmeterprop=3.7/700 # meters per pixel in x dimension
        self.Ypixmeterprop=30/720 # meters per pixel in y dimension
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None 
        # polynomial coeffs of the last n fits of the line
        self.recent_coeffs = []
        self.recent_coeffsw = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        self.best_fitw = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #polynomial coefficients for the last fit
        self.last_fit = np.array([0,0,0], dtype='float')
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None          
        self.currentmeasures=(0,0)       
        self.lastmeasures=(0,0)


# PIPELINE 
def lanepipeline(input_image):
    # 2: UNDISTORD INPUT FRAME
    udtd_image=undistord_frame(input_image,"calibration_results.p")
    # 3: COLOR AND GRADIENT THRESHOLD
    thrld_image=threshold_frame(udtd_image)
    #plot_results(udtd_image, thrld_image)
    # 4: PERSPECTIVE TRANSFORM 
    result_perspective_trans=perspective_transform_frame(thrld_image)
    # 5: DETECT LANE LINES AND  DETERMINE THE LANE CURVATURE
    result=detect_lane_lines(result_perspective_trans[0])
    # 6: PRINT RESULT OVERLAPED CURVE
    process_image=project_lines(input_image,udtd_image,result_perspective_trans[0],result_perspective_trans[1],result[1],result[2],result[3],result[4],result[5],result[6],result[7])
    
    return process_image
    

       
# MAIN PROCESS

# 1: CALIBRATE THE CAMERA
#calibrate_camera() #ONLY FOR FIRST TIME 
# CALL PIPELINE
#create instance of a lane line left and right
L_Lane_Line = Line()
R_Lane_Line = Line()

