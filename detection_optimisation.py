from statistics import mean
import multiprocessing
import threading
import time

import numpy as np
import cv2
from skimage import measure
import imutils
from imutils import contours

VIDEO_SRC = 0


def frame_grabber():
    cap = cv2.VideoCapture(VIDEO_SRC)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
    cap.release()


def rescale(frame, scale: int):
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def process_frame(frame, median_frame):
    dframe = cv2.absdiff(frame, median_frame)
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

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

        if numPixels > 200:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort left to right
    cnts = imutils.grab_contours(cnts)

    buffer = 15

    image_out = np.zeros(frame.shape, np.uint8)
    if cnts:
        cnts = contours.sort_contours(cnts)[0]
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            image_out[
                int(y - buffer) : int(y + h + buffer),
                int(x - buffer) : int(x + w + buffer),
            ] = frame[
                int(y - buffer) : int(y + h + buffer),
                int(x - buffer) : int(x + w + buffer),
            ]

    return image_out, dframe


median_frames = []


def get_windowed_median_frame(f, count: int = 25):
    if len(median_frames) == count:
        median_frames.pop(0)
    median_frames.append(f)
    median_frame = np.median(median_frames, axis=0).astype(dtype=np.uint8)
    return median_frame


def apply_crop(f, roi):
    return f[int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])]


NUMBER_MEDIAN_F = 50
SCALE_FACTOR = 30  # 25% of original


def camera_capture_thread(roi, frames, to_processing_queue):
    index = 0
    for raw_f in frames:
        f = rescale(raw_f, SCALE_FACTOR)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        cropped = apply_crop(f, roi)
        median_frame = get_windowed_median_frame(cropped, NUMBER_MEDIAN_F)
        to_processing_queue.put((index, (cropped, median_frame)))
        index += 1


def render(render_queue):
    frames = {}
    upto = {}
    while True:
        frame_type, frame_id, frame = render_queue.get()

        frame_type_upto = upto.get(frame_type, 0)

        if frame_id == frame_type_upto:
            cv2.imshow(frame_type, frame)
            cv2.waitKey(1)
            upto[frame_type] = frame_type_upto + 1
            continue

        frame_type_frames = frames.get(frame_type, {})
        frame_type_frames[frame_id] = frame

        frames[frame_type] = frame_type_frames

        try:
            exists = frame_type_frames.pop(frame_type_upto)
            cv2.imshow(frame_type, exists)
            cv2.waitKey(1)
            upto[frame_type] = frame_type_upto + 1
        except KeyError:
            pass


def process_worker(in_proc_queue, out_render_queue):
    while True:
        index, (cropped, median_frame) = in_proc_queue.get()
        processed_f, difference = process_frame(cropped, median_frame)
        out_render_queue.put(("processed", index, processed_f))
        # out_render_queue.put(("difference", index, difference))
        out_render_queue.put(("cropped", index, cropped))
        # out_render_queue.put(("median_frame", index, median_frame))


COUNT_PROCESSES = 2


def main():

    frames = frame_grabber()
    raw_f = next(frames)
    f = rescale(raw_f, SCALE_FACTOR)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    region = cv2.selectROI(f)

    PROCESS_CTX = multiprocessing.get_context("fork")  # change to fork if not on darwin

    to_proc_queue = PROCESS_CTX.Queue(maxsize=8)
    to_render_queue = PROCESS_CTX.Queue(maxsize=8)

    capture_thread = threading.Thread(
        target=camera_capture_thread, args=(region, frames, to_proc_queue,)
    )
    capture_thread.start()

    process_processes = [
        PROCESS_CTX.Process(
            target=process_worker, args=(to_proc_queue, to_render_queue)
        )
        for _ in range(2)
    ]

    for proc in process_processes:
        proc.start()

    render(to_render_queue)


if __name__ == "__main__":
    main()
