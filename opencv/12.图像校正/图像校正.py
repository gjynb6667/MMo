import  cv2
import  numpy as  np
image = cv2.imread('test1.png')
image_shape = image.shape
points = np.float32([[200,100],[700,150],[140,400],[650,460]])
point2 = np.float32([[0,0],[image_shape[1],0],[0,image_shape[0]],[image_shape[1],image_shape[0]]])
cv2.line(image,points[0].astype(np.int64).tolist(),points[1].astype(np.int64).tolist(),color=(0,0,255),lineType=cv2.LINE_AA)
cv2.line(image,points[0].astype(np.int64).tolist(),points[2].astype(np.int64).tolist(),color=(0,0,255),lineType=cv2.LINE_AA)
cv2.line(image,points[2].astype(np.int64).tolist(),points[3].astype(np.int64).tolist(),color=(0,0,255),lineType=cv2.LINE_AA)
cv2.line(image,points[1].astype(np.int64).tolist(),points[3].astype(np.int64).tolist(),color=(0,0,255),lineType=cv2.LINE_AA)
M = cv2.getPerspectiveTransform(points,point2)
image_warpPerspective = cv2.warpPerspective(image,M,(image_shape[1],image_shape[0]))
cv2.imshow('image',image)
cv2.imshow('warpPerspective',image_warpPerspective)
cv2.waitKey(0)