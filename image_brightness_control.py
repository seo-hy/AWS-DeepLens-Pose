import os
import cv2
import numpy as np

input_dir = input("input_dir : ")
output_dir = input("output_dir : ")
if output_dir not in os.listdir():
	os.mkdir(output_dir)
keyword= input("keyword : ")
cnt = input("startnum : " )
cnt = int(cnt)
images = os.listdir(input_dir) 
for image in images:
	img = cv2.imread(input_dir+"/"+image)
	val = 100
	array = np.full(img.shape,(val,val,val),dtype=np.uint8)

	
	#set
	add_dst = cv2.add(img, array)
	sub_dst = cv2.subtract(img, array)
	
	#show
	#cv2.imshow('img',img)
	#cv2.imshow('add',add_dst)	
	#cv2.imshow('sub',sub_dst)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	
	cv2.imwrite(output_dir+"/"+keyword+"add"+ str(cnt)+".jpg",add_dst)
	cv2.imwrite(output_dir+"/"+keyword+"sub"+ str(cnt)+".jpg",sub_dst)
	cnt = cnt+1
	#if cnt ==100:
	#	break
