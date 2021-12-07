from collections import deque
from stream_listener import StreamListener

STREAM_URL = "http://localhost:8080"
REATTEMPT_INTERVAL = 2

frame_queue = deque(maxlen=30)

stream_listener = StreamListener(STREAM_URL, REATTEMPT_INTERVAL)
stream_listener.read_stream(frame_queue)