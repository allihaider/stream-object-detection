import os
import time
from collections import deque
from stream_listener import StreamListener
# from multiprocessing import Process
import threading

STREAM_URL = "http://localhost:8080"
REATTEMPT_INTERVAL = 2

frame_queue = deque(maxlen=30)
stream_listener = StreamListener(STREAM_URL, REATTEMPT_INTERVAL)
stream_listening_thread = threading.Thread(target=stream_listener.read_stream, args=(frame_queue,))
stream_listening_thread.start()

while True:
    print(f"{threading.get_ident()}:{len(frame_queue)}")
    time.sleep(2)