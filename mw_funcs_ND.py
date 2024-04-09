
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage import io
import skimage
import sys
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

def ViewDataset(img):
    
    """ View the Ultrasound image
    
    The image will also show you a colour bar showing the intensity of the image
    """

    # Show the Image ----- :
    fig,ax = plt.subplots(ncols=2,figsize=(15,5))
    a=ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Image")
    ax[0].set_ylabel("H [pixels]")
    ax[0].set_xlabel("W [pixels]")
    fig.colorbar(a,ax=ax[0])

    # Show the Histogram ----- :
    ax[1].set_xlabel("Intensity")
    ax[1].set_ylabel("pixel count")
    ax[1].set_title("Image Intensity")
    ax[1].set_xlim([0.0, 255.0])  # <- named arguments do not work here
    #ax[1].plot(bin_edges[0:-1], histogram) 
    ax[1].hist(img.reshape(-1, np.shape(img)[0]*np.shape(img)[1])[0,:],bins=50)
    ax[1].set_xlim([0.0, 255.0]) 
    plt.tight_layout()
    plt.show()
    
def ViewEdges(img, edges):
    fig,ax = plt.subplots(ncols=2,figsize=(15,5))
    a=ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Image")
    ax[0].set_ylabel("H [pixels]")
    ax[0].set_xlabel("W [pixels]")
    fig.colorbar(a,ax=ax[0])
    
    ax[1].imshow(img, cmap='gray')
    ax[1].imshow(edges, cmap='viridis', alpha=0.5)
    ax[1].set_title("Edges")
    ax[1].set_ylabel("H [pixels]")
    ax[1].set_xlabel("W [pixels]")
    plt.tight_layout()
    plt.show()
    
def showcircles(hough_res, hough_radii, image,num_circs=5):
  # Select the most prominent 3 circles
  accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=num_circs)

  # Draw them
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
  image = color.gray2rgb(image)
  for center_y, center_x, radius in zip(cy, cx, radii):
      circy, circx = circle_perimeter(center_y, center_x, radius,
                                      shape=image.shape)
      image[circy, circx] = (220, 20, 20)

  ax.imshow(image, cmap=plt.cm.gray)

  return radii


def ViewZoom(img,x,y,L=20):
    
    """ View zoomed region in more detail
    
    The image will also show you a colour bar showing the intensity of the image
    """

    import matplotlib.patches as patches
    # Show the Image ----- :

    X,Y =np.shape(img)

    if x>(X-L):
        print(f"\n **WARNING**: zoom coordinate outside the size of the image. Image size: W = {X}, H = {Y} \n") 
    elif x<0:
        print(f"\n **WARNING**: zoom coordinate outside the size of the image. Image size: W = {X}, H = {Y} \n")
    elif y<0:
        print(f"\n **WARNING**: zoom coordinate outside the size of the image. Image size: W = {X}, H = {Y} \n")   
    elif y>Y-L:
        print(f"\n **WARNING**: zoom outside the size of the image. Image size: W = {X}, H = {Y} \n")
        
    else:

        fig,ax = plt.subplots(ncols=3,figsize=(20,5))
        a=ax[0].imshow(img, cmap='gray',vmin=0,vmax=255)

        # Add the zoom patch
        rect = patches.Rectangle((x,y), L, L, linewidth=2, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

        ax[0].hlines(y,xmin=0,xmax=x,linestyles="dashed",colors='r')
        ax[0].vlines(x,ymin=y,ymax=Y-5,linestyles="dashed",colors='r')

        ax[0].set_title("Whole Image")
        ax[0].set_ylabel("H [pixels]")
        ax[0].set_xlabel("W [pixels]")
        fig.colorbar(a,ax=ax[0])

        img_zoom = img[x:x+L,y:y+L]
        img_zoom = img[y:y+L,x:x+L]
        a=ax[1].imshow(img_zoom, cmap='gray',vmin=0,vmax=255)
        ax[1].set_title("Zoomed Image")
        ax[1].set_ylabel("H [pixels]")
        ax[1].set_xlabel("W [pixels]")
        ax[1].set_xticks(np.arange(0, L, step=2, dtype=int))
        ax[1].set_yticks(np.arange(0, L, step=2, dtype=int))
        fig.colorbar(a,ax=ax[1])


        # Show the Histogram ----- :
        ax[2].set_xlabel("Intensity")
        ax[2].set_ylabel("pixel count")
        ax[2].set_title("Zoomed Image Intensity")
        ax[2].set_xlim([0.0, 255.0]) 
        ax[2].hist(img_zoom.reshape(-1, np.shape(img_zoom)[0]*np.shape(img_zoom)[1])[0,:],bins=10)


        plt.tight_layout()
        plt.show()

def ThresholdImage(img, T):
    """Apply a threshold of T to our image

    Args:
        img (array): the image we want to threshold
        T (int): the value of the threshold
    """

    if T<0 or T>255:
        print("ERROR: T must be between 0 and 255")
        sys.exit()

    else:
        return np.where(img<T,0,1)


def BandwidthThresholdImage(img, Ti,Tf):
    """Apply a threshold of T to our image

    Args:
        img (array): the image we want to threshold
        T (int): the value of the threshold
    """


    if Ti<0 or Ti>255:
        print("ERROR: T must be between 0 and 255")
        sys.exit()

    elif Tf<0 or Tf>255:
        print("ERROR: T must be between 0 and 255")
        sys.exit()
    else:
        return np.where((img>Ti) & (img<Tf),1,0)

def BandWidthViewThreshold(img,Timg,Ti,Tf):
    
    """ View the Ultrasound image
    
    The image will also show you a colour bar showing the intensity of the image
    """

    # Show the Image ----- :
    fig,ax = plt.subplots(ncols=3,figsize=(20,5))
    a=ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Image")
    ax[0].set_ylabel("H [pixels]")
    ax[0].set_xlabel("W [pixels]")
    fig.colorbar(a,ax=ax[0])

    # Show the Histogram ----- :
    ax[1].set_xlabel("Intensity")
    ax[1].set_ylabel("pixel count")
    ax[1].set_title("Image Intensity")
    ax[1].hist(img.reshape(-1, np.shape(img)[0]*np.shape(img)[1])[0,:],bins=50)
    a=ax[1].axvline(x=Ti,color='r', linestyle = '--',label='Lower Threshold (Ti)')
    a=ax[1].axvline(x=Tf,color='g', linestyle = '--',label='Upper Threshold (Tf)')
    ax[1].legend()

    a=ax[2].imshow(Timg, cmap='gray',vmin=0,vmax=1)
    ax[2].set_title("Thresholded Image")
    ax[2].set_ylabel("H [pixels]")
    ax[2].set_xlabel("W [pixels]")
    fig.colorbar(a,ax=ax[2])
    plt.tight_layout()
    plt.show()


def ViewThreshold(img,Timg,T):
    
    """ View the Ultrasound image
    
    The image will also show you a colour bar showing the intensity of the image
    """

    # Show the Image ----- :
    fig,ax = plt.subplots(ncols=3,figsize=(20,5))
    a=ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Image")
    ax[0].set_ylabel("H [pixels]")
    ax[0].set_xlabel("W [pixels]")
    fig.colorbar(a,ax=ax[0])

    # Show the Histogram ----- :
    ax[1].set_xlabel("Intensity")
    ax[1].set_ylabel("pixel count")
    ax[1].set_title("Image Intensity")
    ax[1].hist(img.reshape(-1, np.shape(img)[0]*np.shape(img)[1])[0,:],bins=50)
    ax[1].set_xlim([0.0, 255.0]) 
    a=ax[1].axvline(x=T,color='r', linestyle = '--',label='Threshold (T)')
    ax[1].legend()

    a=ax[2].imshow(Timg, cmap='gray',vmin=0,vmax=1)
    ax[2].set_title("Thresholded Image")
    ax[2].set_ylabel("H [pixels]")
    ax[2].set_xlabel("W [pixels]")
    fig.colorbar(a,ax=ax[2])
    plt.tight_layout()
    plt.show()
    

def ShowHoughLines(threshold_image,image,lines,ext=1):

    
    
    
    if ext:
        fig,ax = plt.subplots(ncols=3,figsize=(15,5))
        a=ax[0].imshow(threshold_image, cmap='gray')
        ax[0].set_title("Thresholded Image")
        ax[0].set_ylabel("H [pixels]")
        ax[0].set_xlabel("W [pixels]")
        #fig.colorbar(a,ax=ax[0])

        a=ax[1].imshow(threshold_image, cmap='gray')
        ax[1].set_title("Tresholded Image with Hough Lines")
        ax[1].set_ylabel("H [pixels]")
        ax[1].set_xlabel("W [pixels]")

        for i,line in enumerate(lines):
            p0, p1 = line
            ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]),label=f'Line {i}')
        
        ax[2].legend()
        a=ax[2].imshow(image, cmap='gray')
        ax[2].set_title("Image with Hough Lines")
        ax[2].set_ylabel("H [pixels]")
        ax[2].set_xlabel("W [pixels]")

        for i,line in enumerate(lines):
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]),label=f'Line {i}')
        ax[2].legend()
        
        
  
    else:
        fig,ax = plt.subplots(ncols=2,figsize=(10,5))
        a=ax[0].imshow(threshold_image, cmap='gray')
        ax[0].set_title("Image")
        ax[0].set_ylabel("H [pixels]")
        ax[0].set_xlabel("W [pixels]")
        #fig.colorbar(a,ax=ax[0])

        a=ax[1].imshow(threshold_image, cmap='gray')
        ax[1].set_title("Image with Hough Lines")
        ax[1].set_ylabel("H [pixels]")
        ax[1].set_xlabel("W [pixels]")

        for i,line in enumerate(lines):
            p0, p1 = line
            ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]),label=f'Line {i}')
        

    plt.show()

def ComputeLength(lines):
    """ Loop through

    Args:
        lines (_type_): _description_
    """    
    
    for i,line in enumerate(lines):
        p0, p1 = line
        L = ((p0[0]-p1[0])**2 + (p0[1]- p1[1])**2)**0.5

        print(f"Length of line {i} = {L:.1f} pixels")

def FemurLength():
    """ Make a plot of gestational age vs femur lengh

    """

    # Gestational age in weeks
    gw = np.arange(14,40)
    # femur length
    femur_len =-39.9616 + 4.32298*gw -0.0380156*(gw**2)
    #sd = np.exp(0.605843-42.0014*gw**(1/2) +0.00000917972 *gw**3)
    #upper = femur_len+sd*3
    #lower = femur_len-sd*3

    plt.figure()
    plt.plot(gw,femur_len,'g')
    #plt.plot(gw,upper,'g--')
    #plt.plot(gw,lower,'g--')
    plt.xlabel("Age (week)")
    plt.ylabel("Femur Length (mm)")
    plt.grid()
    plt.show()

def FindFetalAge(length):

    if length >70 or length<10:
        print("\n ****WARNING****: Femur length outside the usual range, try again \n")
    else:
        # Gestational age in weeks
        gw = np.arange(14,40)
        femur_len =  -39.9616 + 4.32298*gw -0.0380156*(gw**2)

        L = min(femur_len, key=lambda x:abs(x-length)) # find the closest length to ours
        ag =gw[list(femur_len).index(L)]
    
        plt.figure()
        plt.plot(gw,femur_len,'g')
        plt.xlabel("Age (week)")
        plt.ylabel("Femur Length (mm)")
        plt.grid()
        plt.plot(ag,L,'rx',label=f"Age = {ag} weeks")
        plt.legend()
        plt.show()

def HeadCircum():
    """ Make a plot of gestational age vs femur lengh

    """

    # Gestational age in weeks
    gw = np.arange(14,40)
    # femur length
    head_circm =-28.2849+1.69267*(gw**2)-0.397485*(gw**2)*np.log(gw)

    plt.figure()
    plt.plot(gw,head_circm,'g')
    plt.xlabel("Age (week)")
    plt.ylabel("Head Circumference (mm)")
    plt.grid()
    plt.show()

def FindFetalAge_head(length):
    if length <70 or length>400:
        print("\n ****WARNING****: Head Circumference outside the usual range, try again \n")
    else:
        # Gestational age in weeks
        gw = np.arange(14,40)
        head_circm =-28.2849+1.69267*(gw**2)-0.397485*(gw**2)*np.log(gw)

        L = min(head_circm, key=lambda x:abs(x-length)) # find the closest length to ours
        ag =gw[list(head_circm).index(L)]
    
        plt.figure()
        plt.plot(gw,head_circm,'g')
        plt.xlabel("Age (week)")
        plt.ylabel("Head Circumference (mm)")
        plt.grid()
        plt.plot(ag,L,'rx',label=f"Age = {ag} weeks")
        plt.legend()
        plt.show()     


def HoughLines(threshold_image,l_min):
    l_min =int(l_min)
    return skimage.transform.probabilistic_hough_line(threshold_image,line_length = l_min)