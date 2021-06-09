import os
from PIL import Image

input_dir = input("input_dir : ")
output_dir = input("output_dir : ")

if output_dir not in os.listdir():
    os.mkdir(output_dir)

files = os.listdir(input_dir)

keyword= input("keyword : ")
cnt = input("startnum : " )
cnt = int(cnt)

#set image size
stream_x = 2688
stream_y = 1520
img_size = 1500

x_size = 0
y_size = 0
x_offset=0
y_offset=0


for img_file in files:
	splt = img_file.split(".")
	file_name = splt.pop()
	if file_name in "jpg jpeg png bmp JPG JPEG PNG BMP":
		image = Image.open(input_dir + "/" + img_file)
		x, y = image.size
		if x >= y :
			new_size = x
			x_offset = 0
			y_offset = int((x-y)/2)
		elif y > x:
			new_size = y
			x_offset = int((y-x) / 2)
			y_offset = 0
	
	#set background color
	background_color = "black"  #white, black ...
	
	#square
	img_square = Image.new("RGB", (new_size, new_size), background_color)
	img_square.paste(image, (x_offset, y_offset))
	img_square = img_square.resize((img_size,img_size))

	#DeepLens Stream size	
	img_stream_size = Image.new("RGB", (stream_x, stream_y), background_color)
	x_offset = (stream_x - img_size)/2
	y_offset = (stream_y - img_size)/2
	img_stream_size.paste(img_square, (int(x_offset), int(y_offset)))
	
	#save	
	outfile_name = keyword+str(cnt)+ ".jpg"
	cnt = cnt+1
	img_stream_size.save(output_dir+"/"+outfile_name)
