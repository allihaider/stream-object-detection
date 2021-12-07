from collections import deque
from stream_listener import StreamListener
from multiprocessing import Process
import time

STREAM_URL = "http://localhost:8080"
REATTEMPT_INTERVAL = 2

frame_queue = deque(maxlen=30)

stream_listener = StreamListener(STREAM_URL, REATTEMPT_INTERVAL)
stream_listening_process = Process(target=stream_listener.read_stream, args=(frame_queue,))
stream_listening_process.start()

while True:
    print(len(frame_queue))
# time.sleep(10)

# stream_listener.read_stream(frame_queue)