import os
from threading import Timer
import scipy
import time
import numpy as np
import awscam
import cv2
import json
import requests
import shutil
import os
import mxnet as mx
from threading import Thread
from scipy.ndimage.filters import gaussian_filter
import PIL.Image as pilimg
import csv

ret, frame = awscam.getLastFrame()
ret,jpeg = cv2.imencode('.jpg', frame) 
Write_To_FIFO = True
class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)
 
    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path,'w')
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue  

                
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    print("image",image.shape)
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h==184) else 184-h # down
    pad[3] = 0 if (w==184) else 184-w # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def greengrass_infinite_infer_run():
    #try:
	modelPath = "models/faster_184.xml"
	results_thread = FIFO_Thread()
	results_thread.start()
	data_file_name = "data.csv"
	datafile = open(data_file_name,'a')
	wr = csv.writer(datafile)
        
	# Load model to GPU (use {"GPU": 0} for CPU)
	mcfg = {"GPU": 1}
	
	model = awscam.Model(modelPath, mcfg)
	doInfer = True
	game_count = 0
	poses = []
	collect_data = True
	
	input_dir = "data/test"
	data_images = os.listdir(input_dir) 
	
	for data_image in data_images:
	    ret, frame = awscam.getLastFrame()
	    if ret == False:
	    	raise Exception("Failed to get frame from the stream")
	    img = pilimg.open(input_dir+"/"+data_image)
	    pix = np.array(img)
	    img =cv2.cvtColor(pix, cv2.COLOR_RGB2BGR)
	    frame = img
	    
	    
	    center = frame.shape[1]/2
	    left = center - (frame.shape[0]/2)
	    scale = frame.shape[0]/184
	    offset = (frame.shape[1] - frame.shape[0]) / 2
	    
	    cframe = frame[0:1520,left:left+1520,:]
	   
	    scaledImg = image_resize(cframe, width=184)
	    heatmap_avg = np.zeros((scaledImg.shape[0], scaledImg.shape[1], 16))
	    paf_avg = np.zeros((scaledImg.shape[0], scaledImg.shape[1], 28))

	    imageToTest = cv2.resize(scaledImg, (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
	    imageToTest_padded, pad = padRightDownCorner(imageToTest, 8, 128)
	    transposeImage = np.transpose(np.float32(imageToTest_padded[:,:,:]), (2,0,1))/255.0-0.5


	    output = model.doInference(transposeImage)

	    h = output["Mconv7_stage4_L2"]
	    p = output["Mconv7_stage4_L1"]
	    heatmap1 = h.reshape([16,23,23]) 
	    heatmap = np.transpose(heatmap1, (1,2,0))


	    heatmap = cv2.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
	    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
	    heatmap = cv2.resize(heatmap, (scaledImg.shape[1], scaledImg.shape[0]), interpolation=cv2.INTER_CUBIC)
	    heatmap_avg = heatmap_avg + heatmap / 1

	    paf1 = p.reshape([28,23,23])
	    paf = np.transpose(paf1, (1,2,0))
	    paf = cv2.resize(paf, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
	    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
	    paf = cv2.resize(paf, (scaledImg.shape[1], scaledImg.shape[0]), interpolation=cv2.INTER_CUBIC)

	    paf_avg = paf_avg + paf / 1

	    msg = "{"
	    probNum = 0 
	    font = cv2.FONT_HERSHEY_SIMPLEX
	    global jpeg

	    dst = scaledImg
	    dst[:,:,2] = dst[:,:,2]+ (heatmap_avg[:,:,15]+0.5)/2*255
	    
	    param={}
	    param['octave'] = 3
	    param['use_gpu'] = 1
	    param['starting_range'] = 0.8
	    param['ending_range'] = 2
	    param['scale_search'] = [0.5, 1, 1.5, 2]
	    param['thre1'] = 0.1
	    param['thre2'] = 0.05
	    param['thre3'] = 0.5
	    param['mid_num'] = 4
	    param['min_num'] = 10
	    param['crop_ratio'] = 2.5
	    param['bbox_ratio'] = 0.25
	    param['GPUdeviceNumber'] = 3

	    all_peaks = []
	    peak_counter = 0

	    for part in range(16):
	    	x_list = []
	    	y_list = []
	    	map_ori = heatmap_avg[:,:,part]
	    	map = gaussian_filter(map_ori, sigma=3)

	    	map_left = np.zeros(map.shape)
	    	map_left[1:,:] = map[:-1,:]
	    	map_right = np.zeros(map.shape)
	    	map_right[:-1,:] = map[1:,:]
	    	map_up = np.zeros(map.shape)
	    	map_up[:,1:] = map[:,:-1]
	    	map_down = np.zeros(map.shape)
	    	map_down[:,:-1] = map[:,1:]

	    	peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
	    	peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])
	    	peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
	    	id = range(peak_counter, peak_counter + len(peaks))
	    	peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

	    	all_peaks.append(peaks_with_score_and_id)
	    	peak_counter += len(peaks)
	    print("all_peaks :",all_peaks)
	    features = []
	    noperson = False
	    count = 0
	    for f in all_peaks:
	    	if count == 15:
	    		break
	    	count = count + 1
	    	if f == []:
	    		noperson = True
	    		break
	    	features.append([f[0][0],f[0][1]])
	    
	    if noperson:
	    	print("No Person Found in Image")
	    	cv2.putText(frame, "No Person", (20,150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 3)
		ret,jpeg = cv2.imencode('.jpg', frame)
	    else:	
	    	pose = np.asarray(features)
	    	headsize = pose[1][1]-pose[0][1]
	    	
	    	#set point
	    	shift = (pose[14][0],pose[14][1])
	    	for i in range(15):
	    		pose[i][0] = pose[i][0] - shift[0]
	    		pose[i][1] = pose[i][1] - shift[1]
		    
		
	    	pose = 1.0*pose/headsize
		#print("features",features)
		#print("pose",pose)
		
	    	pose = list(np.asarray(pose).reshape([15*2]))

	    	#print("pose:",pose)


	    	wr.writerow(pose)
	    	            
	    	color = (0,0,0)
	    	cv2.putText(frame, data_image, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

	    	for i in range(15):
	    		cv2.circle(frame, (features[i][0]*scale+offset,features[i][1]*scale), 20, (0,0,255), thickness=-10)
	    	cv2.circle(frame, (features[7][0]*scale+offset,features[7][1]*scale), 20, (0,0,0), thickness=-10)
	    	cv2.circle(frame, (features[10][0]*scale+offset,features[10][1]*scale), 20, (255,0,0), thickness=-10)
	    	ret,jpeg = cv2.imencode('.jpg', frame)
	    #except Exception as e:
	    #    msg = "Test failed: " + str(e)
	    #    print msg
	datafile.close()
	exit(100)




greengrass_infinite_infer_run()


def function_handler(event, context):
    return
