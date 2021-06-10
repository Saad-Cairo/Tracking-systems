import sys
import time
import numpy as np_

from collections import deque
from Frame_manager import *
import torch
from output_saver import *
from utils import *
import glob
from config_helper import load_config
from custom import Custom
from load_helper import load_pretrain
from test import siamese_init, siamese_track

id_gen = ID_generator()
current_frame_to_be_tracked = None
history_queue = deque(maxlen=2)


def select_active_boxes(current_frame: Frame, previous_frames_queue, thresh):
    boxes = []
    if len(current_frame.boxes) == 0:
        return boxes

    for current_box_ in current_frame.boxes:
        exist = False
        for previous_frame in previous_frames_queue:
            for previous_box in previous_frame.boxes:
                if box_box_iou(current_box_, previous_box) > thresh:
                    current_box_.ID = previous_box.ID
                    exist = True
                    break
            if exist:
                break

        if exist:
            continue
        else:
            current_box_.assign_ID(id_gen)
            boxes.append(current_box_)

    return boxes


def select_matching_box(polygon, frame):
    max_ = 0
    box_ = None
    for frame_box in frame.boxes:
        iou = box_polygon_iou(frame_box, polygon)
        if iou > max_:
            max_ = iou
            box_ = frame_box

    if max_ < 0.8:  # 0.8
        return convert_polygon_to_box(polygon)

    return box_


def get_detection_output_as_frames_generator():
    """
    this function to read the detection csv file.

    """
    detection_df = pd.read_csv('/home/saad/Root/datasets/tracking/helicopter_tracking.csv')
    images_path = "/home/saad/Root/datasets/vot2017/helicopter-set"
    detection_df = detection_df.sort_values("time_step")
    data_len = len(detection_df.index)
    for i in range(data_len):
        idx, x1, y1, x2, y2, type = detection_df.loc[i].to_numpy()
        current_index = idx
        frame_to_send = Frame(idx)
        image_path = join(images_path, str(idx).zfill(8) + '.jpg')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_to_send.set_img(img)
        coordinates = (x1, y1, x2, y2)
        box = Box(coordinates, type)
        frame_to_send.add_box(box)
        next_index = detection_df.loc[i + 1].to_numpy()[0]
        while next_index == current_index:
            idx, x1, y1, x2, y2, type = detection_df.loc[i + 1].to_numpy()
            coordinates = (x1, y1, x2, y2)
            box = Box(coordinates, type)
            frame_to_send.add_box(box)
            i += 1
            next_index = detection_df.loc[i + 1].to_numpy()[0]
        i += 1
        yield frame_to_send, (data_len - i) <= 0


class TrackingManager:

    def __init__(self):
        resume = '/home/saad/Root/vision/Computer_Vision/Tracking-systems/SiamMask_DAVIS.pth'
        config = '/home/saad/Root/vision/Computer_Vision/Tracking-systems/config_davis.json'
        self.cfg = load_config(config=config)
        self.siammask = Custom(anchors=self.cfg['anchors'])
        self.siammask = load_pretrain(self.siammask, resume)

        self.active_boxes = []
        self.frames_generator = get_detection_output_as_frames_generator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.siammask.eval().to(self.device)
        self.tracker_initialized = False
        self.state = None
        self.times = np_.asarray([])

    def track(self):

        current_frame, final_one = next(self.frames_generator)
        if final_one:
            yield 1

        s_ = time.time()
        init_boxes = select_active_boxes(current_frame, history_queue, 0.1)
        targets = []
        for box_ in init_boxes:
            target_pos = np.array([box_.x1 + box_.w / 2, box_.y1 + box_.h / 2])
            target_sz = np.array([box_.w, box_.h])
            # print("x1 = {}, y1 = {}, w = {}, h = {}".format(box.x1, box.y1, box.w, box.h))
            s = {"target_pos": target_pos, "target_sz": target_sz, "x": box_.x1, "y": box_.y1, "w": box_.w,
                 "h": box_.h}
            targets.append(s)
        self.active_boxes.extend(init_boxes)

        if len(init_boxes) > 0:
            if self.state is not None:
                targets.extend(self.state['targets'])
            self.state = siamese_init(current_frame.img, self.siammask, self.cfg['hp'], device=self.device,
                                      targets=targets)  # init tracker
            self.tracker_initialized = True

        if self.tracker_initialized and self.state is not None and len(self.state['targets']) > 0:
            self.state = siamese_track(self.state, current_frame.img)
            t = 0
            while t < len(self.state['targets']):
                # check that the tracked object still exist
                score = self.state['targets'][t]['score']
                if score <= .001:
                    print("remove box because its score is {}".format(self.state['targets'][t]['score']))
                    self.remove_gone_boxes(self.state['targets'][t])
                    del self.state['targets'][t]
                    continue

                target = self.state['targets'][t]

                boxx = select_matching_box(target['ploygon'], current_frame)
                self.state['targets'][t]['ploygon'] = [[boxx.x1, boxx.y1], [boxx.x1, boxx.y2], [boxx.x2, boxx.y2],
                                                       [boxx.x2, boxx.y1]]
                # assign ID to the tracked object
                x, y, w, h = target['x'], target['y'], target['w'], target['h']
                for o, active_box in enumerate(self.active_boxes):
                    if active_box.x1 == x and active_box.y1 == y and active_box.w == w and active_box.h == h:
                        boxx.ID = active_box.ID
                        boxx.type = active_box.type

                # frame.get_coord_depend_seg(mask,boxx.x1,boxx.y1,boxx.x2,boxx.y2, id)
                cv2.rectangle(current_frame.img, (int(boxx.x1), int(boxx.y1)), (int(boxx.x2), int(boxx.y2)),
                              (255, 0, 0), 2)
                center = [int(x) for x in target['target_pos']]
                cv2.putText(current_frame.img, str(boxx.ID), tuple(center), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                t += 1
                # frame.add_polygon(target['ploygon'], id)
                current_frame.add_box(boxx)

                history_queue.append(current_frame)
            print(f"tracked in {time.time() - s_}")
            self.times = np_.append(self.times, [time.time() - s_], axis=0)
            cv2.putText(current_frame.img, "current frame", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
            current_frame.img = cv2.cvtColor(current_frame.img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"/home/saad/Root/datasets/tracking/tracking_facepass_case/{current_frame.frame_indx}.jpg",
                        current_frame.img)
            yield current_frame

    def remove_gone_boxes(self, target):
        x, y, w, h = target['x'], target['y'], target['w'], target['h']
        i = 0
        for active_box in self.active_boxes:
            if active_box.x1 == x and active_box.y1 == y and active_box.w == w and active_box.h == h:
                del self.active_boxes[i]
                break
            i += 1

    def get_tracker_fps(self):
        avg = np_.average(self.times)
        return 1 / avg, avg


if __name__ == "__main__":
    tm = TrackingManager()
    while True:
        tracked = next(tm.track())
        if tracked == 1:
            break
    print(tm.get_tracker_fps())
