from statistics import mean
from multiprocessing import Queue, Process
import time

import numpy as np
import cv2
from skimage import measure
import imutils
from imutils import contours

def frame_grabber():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
    cap.release()

def rescale(frame, scale: int):
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
    return frame

SCALE_FACTOR = 50

def camera_capture(frame_out: Queue):
    frame_count = 0
    last_frame = time.time()
    grabber = frame_grabber()
    while True:
        frame = next(grabber)
        frame_out.put((frame_count, rescale(frame,SCALE_FACTOR)))
        frame_count += 1
        now = time.time()
        if frame_count % 100 == 0:
            print(f'capture fps: {round(1 / (now - last_frame))}')
        last_frame=now

def get_median_frame(count: int = 25):
    frames = []
    cap = cv2.VideoCapture(0)
    for fid in range(count):
        # cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(rescale(frame,SCALE_FACTOR))
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cap.release()
    return median_frame

def process_frame(in_queue: Queue, out_queue: Queue, median_frame):
    time_averages = []
    while True:
        frame_count, frame_col = in_queue.get()

        start_time = time.time()
        frame = cv2.cvtColor(frame_col, cv2.COLOR_BGR2GRAY)
        dframe = cv2.absdiff(frame, median_frame)
        th, dframe = cv2.threshold(dframe, 60, 255, cv2.THRESH_BINARY)

        thresh = cv2.erode(dframe, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # connected component analysis
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            # if background label
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            if numPixels > 300:
                mask = cv2.add(mask, labelMask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        # sort left to right
        cnts = imutils.grab_contours(cnts)

        buffer = 0

        image_out = np.zeros(frame_col.shape, np.uint8)
        if cnts:
            cnts = contours.sort_contours(cnts)[0]
            for (i, c) in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(c)
                image_out[int(y - buffer): int(y + h + buffer), int(x- buffer): int(x + w + buffer)] = \
                frame_col[int(y - buffer): int(y + h + buffer), int(x- buffer): int(x + w + buffer)]

        # only for logging averages
        time_averages.append(time.time()-start_time)
        if len(time_averages) == 50:
            av = mean(time_averages)
            print('average time:', mean(time_averages), 'max_fps:', round(1/av))
            time_averages = []

        out_queue.put((frame_count, image_out))

def render_thread(frame_in):
    up_to=0
    frame_store = {}
    while True:
        frame_count, frame = frame_in.get()
        if frame_count == up_to:
            # no need to use store just render that bad boy
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            up_to += 1
            continue

        frame_store[frame_count] = frame
        try:
            frm = frame_store.pop(up_to)
            cv2.imshow('frame', frm)
            cv2.waitKey(1)
            up_to += 1
        except Exception:
            pass

def main():
    # Calculate the median along the time axis
    medianFrame = get_median_frame()
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    # in and out queues (max-size prevents render drift)
    frame_in = Queue(maxsize=8)
    frame_out = Queue(maxsize=8)

    # boot up the threads
    threads = [Process(target=process_frame, args=(frame_in, frame_out, grayMedianFrame)) for i in range(4)]
    threads.append(Process(target=render_thread, args=(frame_out,)))
    threads.append(Process(target=camera_capture, args=(frame_in,)))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Destroy all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
