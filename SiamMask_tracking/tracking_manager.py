import sys

from collections import deque
from Frame_manager import *
import torch
import os

import argparse
import pathlib
from os.path import isfile
from skvideo import io
from output_saver import *
from utils import *
import glob
from config_helper import load_config
from custom import Custom
from load_helper import load_pretrain
from test import siamese_init, siamese_track


class TrackingManager:

    def __init__(self, detection_csv_path, imgs_path, buffer_size, output_coordinator, out_vid_path, sampling_ratio):
        resume = '/home/saad/Root/vision/Computer_Vision/Tracking-systems/SiamMask_DAVIS.pth'
        config = '/home/saad/Root/vision/Computer_Vision/Tracking-systems/config_davis.json'
        self.cfg = load_config(config=config)
        self.siammask = Custom(anchors=self.cfg['anchors'])
        self.siammask = load_pretrain(self.siammask, resume)

        self.detection_csv_path = detection_csv_path
        self.sampling_ratio = sampling_ratio
        self.imgs_path = imgs_path
        self.buffer_size = buffer_size
        self.queue = deque(maxlen=self.buffer_size)
        self.active_boxes = []
        self.updated_boxes = []

        self.fm = FrameManager(detection_csv_path, imgs_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.siammask.eval().to(self.device)
        self.output_coordinator = output_coordinator
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(out_vid_path, fourcc, self.sampling_ratio, (848, 480))

    def track(self):

        self.fm.fill_buffer(self.queue, self.buffer_size)
        # self.fm.get_next_frame(self.queue)
        i = 0
        state = None
        num_of_frames = len(glob.glob(self.imgs_path + '/*.jpg'))
        while True:
            if i == num_of_frames - 1:
                tracks = self.output_coordinator.save_results_in_pickle()
                print('Pickle file is saved')
                return tracks

            print("in frame {} len of active_boxes = {}".format(i, len(self.active_boxes)))
            init_boxes = self.fm.select_active_boxes(self.queue, self.updated_boxes, 0.1, i)
            targets = []
            for box in init_boxes:
                target_pos = np.array([box.x1 + box.w / 2, box.y1 + box.h / 2])
                target_sz = np.array([box.w, box.h])
                # print("x1 = {}, y1 = {}, w = {}, h = {}".format(box.x1, box.y1, box.w, box.h))
                s = {"target_pos": target_pos, "target_sz": target_sz, "x": box.x1, "y": box.y1, "w": box.w, "h": box.h}
                targets.append(s)
            self.active_boxes.extend(init_boxes)
            self.updated_boxes.extend(init_boxes)

            if len(init_boxes) > 0:
                if state is not None:
                    targets.extend(state['targets'])
                print("in frame {} init_boxes={}".format(i, len(init_boxes)))
                state = siamese_init(self.queue[0].img, self.siammask, self.cfg['hp'], device=self.device,
                                     targets=targets)  # init tracker
            if i > 0 and state is not None and len(state['targets']) > 0 and len(self.queue) > 0:
                state = siamese_track(state, self.queue[0].img)

                frame = self.queue[0]
                # print("len of target = {}".format(len(state['targets'])))
                t = 0
                while t < len(state['targets']):
                    # check that the tracked object still exist
                    score = state['targets'][t]['score']
                    if state['targets'][t]['score'] <= .5:
                        print(" reomve box because its score is {}".format(state['targets'][t]['score']))
                        self.remove_gone_boxes(state['targets'][t])
                        del state['targets'][t]
                        continue

                    target = state['targets'][t]

                    boxx = self.select_matching_box(target['ploygon'], frame)
                    state['targets'][t]['ploygon'] = [[boxx.x1, boxx.y1], [boxx.x1, boxx.y2], [boxx.x2, boxx.y2],
                                                      [boxx.x2, boxx.y1]]
                    # assign ID to the tracked object
                    x, y, w, h = target['x'], target['y'], target['w'], target['h']
                    id = -5
                    type = None
                    for o, active_box in enumerate(self.active_boxes):
                        if active_box.x1 == x and active_box.y1 == y and active_box.w == w and active_box.h == h:
                            id = active_box.ID
                            type = active_box.type
                            # self.active_boxes[i] = boxx
                            '''if boxx.ID == -1:
                                print(boxx)
                                print(target['ploygon'])
                                print("convert the polygon to box with ID = {}".format(id))'''
                            boxx.ID = id
                            boxx.type = type
                            self.updated_boxes[o] = boxx

                    # frame.get_coord_depend_seg(mask,boxx.x1,boxx.y1,boxx.x2,boxx.y2, id)
                    cv2.rectangle(frame.img, (int(boxx.x1), int(boxx.y1)), (int(boxx.x2), int(boxx.y2)), (255, 0, 0), 2)
                    center = [int(x) for x in target['target_pos']]
                    cv2.putText(frame.img, str(id), tuple(center), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                    t += 1
                    # frame.add_polygon(target['ploygon'], id)
                    frame.add_updated_box(boxx)
                    # print(boxx)

                cv2.putText(frame.img, "Frame {}".format(i), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
                frame.img = cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB)
                self.writer.write(frame.img)
                self.output_coordinator.add_frame(frame)
            self.fm.get_next_frame(self.queue)
            i += 1

    def remove_gone_boxes(self, target):
        x, y, w, h = target['x'], target['y'], target['w'], target['h']
        i = 0
        for active_box in self.active_boxes:
            if active_box.x1 == x and active_box.y1 == y and active_box.w == w and active_box.h == h:
                del self.active_boxes[i]
                del self.updated_boxes[i]
                break
            i += 1

    def select_matching_box(self, polygon, frame):
        maxx = 0
        box = None
        # print("Matching boxes")
        for frame_box in frame.boxes:
            iou = box_polygon_iou(frame_box, polygon)
            if iou > maxx:
                maxx = iou
                box = frame_box

        if maxx < 0.8:  # 0.8
            return convert_polygon_to_box(polygon)

        return box


if __name__ == "__main__":
    # detection_path = '/media/innolab/3448C50948C4CB36/tested_video_temp/october_bridge/detection_images'
    # detection_csv_file = '/media/innolab/3448C50948C4CB36/tested_video_temp/october_bridge/detections/detections.csv'
    detection_path = '/media/innolab/3448C50948C4CB36/tracking_test/detection_imgs'
    detection_csv_file = '/media/innolab/3448C50948C4CB36/tracking_test/detections.csv'
    output_coordinator = Output_coordinator('/media/innolab/3448C50948C4CB36/tracking_test')
    tm = TrackingManager(detection_csv_file, detection_path, 10, output_coordinator)
    tm.track()
