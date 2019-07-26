from PIL import Image
import os

root_dir = "C:\\Users\\Administrator\\Desktop\\garbage\\data2"
count = 0
for root, dirs, files in os.walk(root_dir):
	for f in files:
		image_path = os.path.join(root, f)
		try:
			Image.open(image_path)
		except Exception as e:
			print("{} removing...".format(image_path))
			count += 1
			os.remove(image_path)
print("error image num: {}".format(count))
		
		

