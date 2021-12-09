import time
import threading
import cv2


class StreamListener:
	def __init__(self, address, reattempt_interval):
		self.address = address
		self.reattempt_interval = reattempt_interval

	def read_stream(self, queue):
		cap = cv2.VideoCapture(self.address)

		while True:
			if (cap.isOpened() == False):
				print(f'{threading.get_ident()}: Could not open stream')

			while(cap.isOpened()):
				ret, frame = cap.read()
				if ret:
					queue.append(frame)
					print(f"{threading.get_ident()}:Frame received")
				else:
					print(f"{threading.get_ident()}:No data received")

			time.sleep(self.reattempt_interval)
