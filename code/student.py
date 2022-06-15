import numpy as np
import cv2
from skimage.filters import prewitt_h,prewitt_v
from skimage.transform import rescale
from skimage.feature import peak_local_max



def get_interest_points(image, alpha=0.06, min_distance=20,thresh=0.1):
   
    R=cv2.cornerHarris(image,4,9,alpha)
    thresh=0.1*R.max()
    x=[]
    y=[]
    for i in range(0,R.shape[0]):
        for j in range(0,R.shape[1]):
            if R[i,j]>=thresh:
                x.append(j)
                y.append(i)
    R[R < thresh*np.max(R)] = 0  # suppress any R value below threshold
    points = peak_local_max(R, min_distance=min_distance)  # apply nonmaximum suppression

    print("number of harris points",len(x))
    #return points[:, 1], points[:, 0]
    return x,y


def ac_get_features(img, feature_width):
    
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    sift = cv2.SIFT.create(nfeatures = 0, nOctaveLayers = 4, contrastThreshold = 0.04, edgeThreshold = 0.1, sigma = 2)
    kp1,dest = sift.detectAndCompute(gray,None)


    #kpA = sift.detect(gray, None)
    #kpA_computed, desA = sift.compute(gray, kpA) 
    

    drawAkp = cv2.drawKeypoints(gray, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("A", drawAkp)

    cv2.waitKey(0)
    x=[]
    y=[]
    for i in kp1:
        x.append(int(i.pt[0]))
        y.append(int(i.pt[1]))
    return dest,np.array(x),np.array(y)
    
def get_features(img, feature_width):
    
    def dog_octave(img,sigma,num_sigma=4):

        temp=img
        k=2*(1/num_sigma)# so the last one has 2*sigma
        octave=[]
        dog_octave=[]
        cornersx=[]
        cornersy=[]
        for i in range(0,num_sigma):

            if i !=0:
                temp=cv2.GaussianBlur(img,(5,5),sigma*i*k)
            tempx,tempy=get_interest_points(temp)
            cornersx.extend(tempx)
            cornersy.extend(tempy)

            octave.append(temp)
            if i!=0:
                dog_octave.append(octave[i-1]-octave[i])


        return dog_octave,cornersx,cornersy


    def dog_pyramid(img,sigma=2,num_sigma=4,num_scale=4):
        dog_pyramid=[]
        temp=img
        cornersx=[]
        cornersy=[]
        for i in range(1,num_scale+1):
            octave,temp_cornersx,temp_cornersy=dog_octave(temp,sigma,num_sigma)
            if i==1:
                cornersx.extend(temp_cornersx)
                cornersy.extend(temp_cornersy)
            else:
                cornersx.extend(temp_cornersx*2*(i-1))
                cornersy.extend(temp_cornersy*2*(i-1))
            dog_pyramid.append(octave)
            temp=rescale(temp,0.5)
        return dog_pyramid,cornersx,cornersy
        

    def check_max_neighbour(dog_pyramid,cornersx,cornersy,feature_width=16):
        max_matrix=[]
        x=[]
        y=[]
        for i in range(0,len(dog_pyramid)):# loop for different scales
            scale=np.array(dog_pyramid[i])

            for j in range(0,scale.shape[0]):# loop for different sigmas
                print(scale[j].shape)
                #tempy,tempx=get_interest_points(scale[j],feature_width)
                for l,k in zip(cornersx,cornersy):# loop for corner values detected by harris as the points have to be corners
                #for l in range(scale[j].shape[0]):
                 #   for k in range(scale[j].shape[1]):
                    l=int(l)
                    k=int(k)
                    if j==0:
                        neighbours=scale[j:j+1,k-1:k+1,l-1:l+1] # neighbour cube
                    else:
                        neighbours=scale[j-1:j+1,k-1:k+1,l-1:l+1] # neighbour cube
                    if scale[0].shape[0]-int(feature_width/2)>=k>=int(feature_width/2) and scale[0].shape[1]-int(feature_width/2)>=l>=int(feature_width/2): # checks if it can be in the image given feature size
                        if (scale[j,k,l]<=np.min(neighbours) or scale[j,k,l]>=np.max(neighbours)) and scale[j,k,l]>=0.03:# check if local minima or maxima also checks if above a certain threshold 
                            m=np.sqrt(((scale[0,k+1,l])-(scale[0,k-1,l]))**2+((scale[0,k,l+1])-(scale[0,k,l-1]))**2) # magnitude of keypoint
                            theta=2*np.pi+np.arctan2(((scale[0,k,l+1])-(scale[0,k,l-1])),((scale [0,k+1,l])-(scale[0,k-1,l])))# angle of keypoint
                            if i==0: 
                                x.append(l)
                                y.append(k)
                                max_matrix.append([l,k,i,m,theta])
                            else:               
                                x.append(l*2**(i))
                                y.append(k*2**(i))
                                max_matrix.append([l*2**(i),k*2**(i),i,m,theta])
        return np.array(max_matrix),x,y
    num_sigma=5
    num_scale=5
    dog_pyramid,cornersx,cornersy=dog_pyramid(img,num_sigma=num_sigma,num_scale=num_scale)

    max_matrix,x,y=check_max_neighbour(dog_pyramid,cornersx,cornersy,feature_width)
    features=np.zeros((max_matrix.shape[0],int(feature_width/4),int(feature_width/4),8))
    count=0
    print(max_matrix.shape)
    for i in max_matrix:        
        k=int(i[1])
        l=int(i[0])

        if i[2]==0:
            temp=img[k-int(feature_width/2):k+int(feature_width/2),l-int(feature_width/2):l+int(feature_width/2)]                      
        else:                                                                                                                          
            tempk=int((k/(2**i[2])))                                                                                                    
            templ=int((l/(2**i[2])))                                                                                                     
            temp=rescale(img,(0.5**i[2]))[tempk-int(feature_width/2):tempk+int(feature_width/2),templ-int(feature_width/2):templ+int(feature_width/2)]          

        dy=prewitt_h(temp)
        dx=prewitt_v(temp)
        grad=np.array(np.sqrt((dx**2)+(dy**2)))
        
        angles=np.array(np.arctan2(dy,dx))
        angles +=np.pi
        #hist, bin_edges = np.histogram(angles, bins= 36, range=(0, 2*np.pi), weights=grad)
        #angles -= bin_edges[np.argmax(hist)]
        #angles+=2*np.pi
        angles_min=np.min(angles)
        angles_max=np.max(angles)

        for j in range(0,int(feature_width/4)):
            for m in range(0,int(feature_width/4)):
                temp_grad=grad[int((j+1)*feature_width/4-2*int(feature_width/8)):int((j+1)*feature_width/4),int((m+1)*feature_width/4-2*int(feature_width/8)):int((m+1)*feature_width/4)]
                temp_angle=angles[int((j+1)*feature_width/4-2*int(feature_width/8)):int((j+1)*feature_width/4),int((m+1)*feature_width/4-2*int(feature_width/8)):int((m+1)*feature_width/4)]                
                features[count,j,m]=np.histogram(temp_angle,bins=8,range=(angles_min,angles_max),weights=temp_grad)[0]
        count+=1
    features= features.reshape((max_matrix.shape[0], -1,)) 
    dividend = np.linalg.norm(features, axis=1).reshape(-1, 1)
    # Rare cases where the gradients are all zeros in the window
    # Results in np.nan from division by zero.
    dividend[dividend == 0 ] = 1
    features = features / dividend
    print(features.shape)
    return features,np.array(x),np.array(y)

def vec_get_features(img, feature_width):
    
    def dog_octave(img,sigma,num_sigma=4):

        temp=img
        k=2*(1/num_sigma)# so the last one has 2*sigma
        octave=[]
        dog_octave=[]
        cornersx=[]
        cornersy=[]
        for i in range(0,num_sigma):

            if i !=0:
                temp=cv2.GaussianBlur(img,(5,5),sigma*i*k)
            tempx,tempy=get_interest_points(temp)
            cornersx.extend(tempx)
            cornersy.extend(tempy)

            octave.append(temp)
            if i!=0:
                dog_octave.append(octave[i-1]-octave[i])


        return dog_octave,cornersx,cornersy


    def dog_pyramid(img,sigma=2,num_sigma=4,num_scale=4):
        dog_pyramid=[]
        temp=img
        cornersx=[]
        cornersy=[]
        for i in range(1,num_scale+1):
            octave,temp_cornersx,temp_cornersy=dog_octave(temp,sigma,num_sigma)
            if i==1:
                cornersx.extend(temp_cornersx)
                cornersy.extend(temp_cornersy)
            else:
                cornersx.extend(temp_cornersx*2*(i-1))
                cornersy.extend(temp_cornersy*2*(i-1))
            dog_pyramid.append(octave)
            temp=rescale(temp,0.5)
        return dog_pyramid,cornersx,cornersy
        

    def check_max_neighbour(dog_pyramid,cornersx,cornersy,feature_width=16):
        max_matrix=[]
        x=[]
        y=[]
        for i in range(0,len(dog_pyramid)):# loop for different scales
            scale=np.array(dog_pyramid[i])

            for j in range(0,scale.shape[0]):# loop for different sigmas
                print(scale[j].shape)
                #tempy,tempx=get_interest_points(scale[j],feature_width)
                for l,k in zip(cornersx,cornersy):# loop for corner values detected by harris as the points have to be corners
                #for l in range(scale[j].shape[0]):
                 #   for k in range(scale[j].shape[1]):
                    l=int(l)
                    k=int(k)
                    if j==0:
                        neighbours=scale[j:j+1,k-1:k+1,l-1:l+1] # neighbour cube
                    else:
                        neighbours=scale[j-1:j+1,k-1:k+1,l-1:l+1] # neighbour cube
                    if scale[0].shape[0]-int(feature_width/2)>=k>=int(feature_width/2) and scale[0].shape[1]-int(feature_width/2)>=l>=int(feature_width/2): # checks if it can be in the image given feature size
                        if (scale[j,k,l]<=np.min(neighbours) or scale[j,k,l]>=np.max(neighbours)) and scale[j,k,l]>=0.1:# check if local minima or maxima also checks if above a certain threshold 
                            m=np.sqrt(((scale[0,k+1,l])-(scale[0,k-1,l]))**2+((scale[0,k,l+1])-(scale[0,k,l-1]))**2) # magnitude of keypoint
                            theta=2*np.pi+np.arctan2(((scale[0,k,l+1])-(scale[0,k,l-1])),((scale [0,k+1,l])-(scale[0,k-1,l])))# angle of keypoint
                            if i==0: 
                                x.append(l)
                                y.append(k)
                                max_matrix.append([l,k,i,m,theta])
                            else:               
                                x.append(l*2**(i))
                                y.append(k*2**(i))
                                max_matrix.append([l*2**(i),k*2**(i),i,m,theta])
        return np.array(max_matrix),x,y
    num_sigma=5
    num_scale=5
    dog_pyramid,cornersx,cornersy=dog_pyramid(img,num_sigma=num_sigma,num_scale=num_scale)

    max_matrix,x,y=check_max_neighbour(dog_pyramid,cornersx,cornersy,feature_width)
    features=np.zeros((max_matrix.shape[0],int(feature_width/4)*int(feature_width/4),8))
    count=0
    print(max_matrix.shape)

    all_dy=prewitt_h(img)
    all_dx=prewitt_v(img)
    all_grad=np.array(np.sqrt((all_dx**2)+(all_dy**2)))
    
    all_angles=np.array(np.arctan2(all_dy,all_dx))
    all_angles +=np.pi
    #hist, bin_edges = np.histogram(angles, bins= 36, range=(0, 2*np.pi), weights=grad)
    #angles -= bin_edges[np.argmax(hist)]
    #angles+=2*np.pi
    angles_min=np.min(all_angles)
    angles_max=np.max(all_angles)

    x_all=[]
    for i in max_matrix:        
        k=int(i[1])
        l=int(i[0])

        if i[2]==0:
            #temp=img[k-int(feature_width/2):k+int(feature_width/2),l-int(feature_width/2):l+int(feature_width/2)] 
            grad=rescale(all_grad,(0.5**i[2]))[k-int(feature_width/2):k+int(feature_width/2),l-int(feature_width/2):l+int(feature_width/2)] 
            angles=rescale(all_angles,(0.5**i[2]))[k-int(feature_width/2):k+int(feature_width/2),l-int(feature_width/2):l+int(feature_width/2)]                      
        else:                                                                                                                          
            tempk=int((k/(2**i[2])))                                                                                                    
            templ=int((l/(2**i[2])))                                                                                                     
            #temp=rescale(img,(0.5**i[2]))[tempk-int(feature_width/2):tempk+int(feature_width/2),templ-int(feature_width/2):templ+int(feature_width/2)]          

            grad=rescale(all_grad,(0.5**i[2]))[tempk-int(feature_width/2):tempk+int(feature_width/2),templ-int(feature_width/2):templ+int(feature_width/2)]
            angles=rescale(all_angles,(0.5**i[2]))[tempk-int(feature_width/2):tempk+int(feature_width/2),templ-int(feature_width/2):templ+int(feature_width/2)]

        x=[]
        features[count]=[np.histogram(angles[int((j+1)*feature_width/4-2*int(feature_width/8)):int((j+1)*feature_width/4),int((m+1)*feature_width/4-2*int(feature_width/8)):int((m+1)*feature_width/4)]
            ,bins=8,range=(angles_min,angles_max),
            weights=grad[int((j+1)*feature_width/4-2*int(feature_width/8)):int((j+1)*feature_width/4),int((m+1)*feature_width/4-2*int(feature_width/8)):int((m+1)*feature_width/4)])[0]
                for j in range(0,int(feature_width/4)) 
                    for m in range(0,int(feature_width/4))]
        count+=1
    features= features.reshape((max_matrix.shape[0], -1,)) 
    dividend = np.linalg.norm(features, axis=1).reshape(-1, 1)
    # Rare cases where the gradients are all zeros in the window
    # Results in np.nan from division by zero.
    dividend[dividend == 0 ] = 1
    features = features / dividend
    print(features.shape)
    return features,np.array(x),np.array(y)


def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """



    # Initialize variables
    matches = []
    confidences = []
    
    # Loop over the number of features in the first image
    for i in range(im1_features.shape[0]):
        # Calculate the euclidean distance between feature vector i in 1st image and all other feature vectors
        # second image
        distances = np.sqrt(((im1_features[i,:]-im2_features)**2).sum(axis = 1))

        # sort the distances in ascending order, while retaining the index of that distance
        ind_sorted = np.argsort(distances)
        # If the ratio between the 2 smallest distances is less than 0.8
        # add the smallest distance to the best matches
        if (distances[ind_sorted[0]] < 0.9 * distances[ind_sorted[1]]):
        # append the index of im1_feature, and its corresponding best matching im2_feature's index
            matches.append([i, ind_sorted[0]])
            confidences.append(1.0  - distances[ind_sorted[0]]/distances[ind_sorted[1]])
          # How can I measure confidence?
    confidences = np.asarray(confidences)
    confidences[np.isnan(confidences)] = np.min(confidences[~np.isnan(confidences)])     

    return np.asarray(matches), confidences
