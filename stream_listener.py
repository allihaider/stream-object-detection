import time
import cv2

class StreamListener:

	def __init__(self, address, reattempt_interval):
		self.address = address
		self.reattempt_interval = reattempt_interval

	def read_stream(self, queue):
		cap = cv2.VideoCapture(self.address)

		while True:
			if (cap.isOpened() == False):
				print('Could not open stream')

			while(cap.isOpened()):
				ret, frame = cap.read()
				if ret:
					queue.append(frame)
				else:
					print("No data received")

			time.sleep(self.reattempt_interval)