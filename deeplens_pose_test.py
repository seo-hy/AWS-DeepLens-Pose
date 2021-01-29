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

#Load Classification Model
feature_count = 15*2
category_count = 3
batch=10

X_pred = mx.nd.zeros((10,feature_count))
Y_pred = Y = mx.nd.empty((10,))

pred_iter = mx.io.NDArrayIter(data=X_pred,label=Y_pred, batch_size=batch)

filename = os.getcwd()+"/models/DeepLens_pose"
sym, arg_params, aux_params = mx.model.load_checkpoint(filename, 500)

new_model = mx.mod.Module(symbol=sym)
new_model.bind(pred_iter.provide_data, pred_iter.provide_label)
new_model.set_params(arg_params, aux_params)

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
        # Load model to GPU (use {"GPU": 0} for CPU)
	mcfg = {"GPU": 1}
	model = awscam.Model(modelPath, mcfg)
	doInfer = True
	game_count = 0
	poses = []
	collect_data = True
	input_dir = "test"
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

	    	peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > 0.1))
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

	    	shift = (pose[14][0],pose[14][1])
	    	for i in range(15):
	    		pose[i][0] = pose[i][0] - shift[0]
	    		pose[i][1] = pose[i][1] - shift[1]
		    
		
	    	pose = 1.0*pose/headsize
		print("features",features)
		print("pose",pose)
	    	pose = list(np.asarray(pose).reshape([15*2]))

	    	pose =np.asarray(pose)
		pose = mx.nd.array(pose)
		print("pose:",pose)               
		X_pred[0] = pose

                print("X_pred :",X_pred)
		print("y_pred :",Y_pred)
                pred_iter = mx.io.NDArrayIter(data=X_pred,label=Y_pred, batch_size=10)

                a = new_model.predict(pred_iter)[0]
		print("pred_iter : ", pred_iter)
                a= list(a.asnumpy())
		print("a : ", a)
                per = max(a)
		p = str(a.index(max(a)))
                print "pred: " + p
		mytext = "" 
                if p == "0":
			mytext = "standing"
		if p == "1":
			mytext = "hi" 
		if p == "2":
			mytext = "hi"
	    	if p == "3":
			mytext = "warrior2"
		if p == "4":
			mytext = "warrior2" 
		if p == "5":
			mytext = "tree"
	    	if p == "6":
	    		mytext = "tree"
		color = (0,0,0)
		if per<0.5:
			color = (0,0,255)
		else:
			color = (255,0,0)

		cv2.putText(frame, mytext, (20,150), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 3)

		cv2.putText(frame, str(int(per*100))+"%", (20,400), cv2.FONT_HERSHEY_SIMPLEX, 5,color, 3)
	    	BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

	    	POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    

	    	for pair in POSE_PAIRS:
	    		partA = pair[0]             # Head
	    		partA = BODY_PARTS[partA]   # 0
	    		partB = pair[1]             # Neck
	    		partB = BODY_PARTS[partB] 
	    	#if pose_ex[partA] and pose_ex[partB]:
	    		#print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
	    		cv2.line(frame, (features[partA][0]*scale+offset,features[partA][1]*scale), (features[partB][0]*scale+offset,features[partB][1]*scale), (0, 255, 0), 3)
                for i in range(15):
                    cv2.circle(frame, (features[i][0]*scale+offset,features[i][1]*scale), 4, (255,0,0), thickness=-1)
            
            ret,jpeg = cv2.imencode('.jpg', frame)
	    #except Exception as e:
	    #    msg = "Test failed: " + str(e)
	    #    print msg

	exit(100)




greengrass_infinite_infer_run()


def function_handler(event, context):
    return
