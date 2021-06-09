import os
import cv2

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
	imgflip = cv2.flip(img,1) # 1 : Flip left and right
	
	#imgflip = cv2.flip(img,0) # 0 : Upside down
	
	cv2.imwrite(output_dir+"/"+keyword+ str(cnt)+".jpg",imgflip)
	cnt = cnt+1
	#if cnt == 100:
	#	break
