import pandas as pd
from Frame import *
from utils import *
from os.path import join
import numpy as np
from ID_generator import *
from collections import deque
#from utils import *
#from box import *
import glob


class FrameManager:
    '''
    this class is implemented to manage the images and detection csv file with bounding boxes,
    so this manager is responsible to load the image with its bounding boxes and compare the boxes
     to merge them.

    '''
    def __init__(self, detection_csv_path, detection_imgs_path):
        self.detection_csv_path = detection_csv_path
        self.detection_imgs_path = detection_imgs_path
        self.detection_data = self.read_detection_output()
        self.last_added_frame_index = 0
        self.last_readed_row = 0
        self.id_gen = ID_generator()
        self.num_of_frames = len(glob.glob(self.detection_imgs_path+'/*.jpg'))

    def read_detection_output(self):
        '''
        this function to read the detection csv file.

        '''
        detection_df = pd.read_csv(self.detection_csv_path)
        detection_df = detection_df.sort_values("time_step")
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        type_list = []
        idx_list = []
        data = []

        for name in detection_df.index:
            idx, x1, y1, x2, y2, type = detection_df.loc[name].to_numpy()
            idx_list.append(idx)
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
            type_list.append(type)
            data.append(tuple([idx, x1, y1, x2, y2, type]))
        return data

    def fill_buffer(self, frame_queue, buffer_size):
        '''
        this function is to fill the buffer with initial frames.

        '''
        prev_idx = self.detection_data[0][0]
        j = 0
        t = buffer_size
        frame = None
        while t > 0:

            coordinates = (self.detection_data[j][1], self.detection_data[j][2],
                   self.detection_data[j][3], self.detection_data[j][4])

            if(self.detection_data[j][0] != prev_idx):
                image_path = join(self.detection_imgs_path, str(prev_idx).zfill(8)+'.jpg')
                # image_path = join(self.detection_imgs_path, str(prev_idx).zfill(8))
                # print(image_path)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame.set_img(img)
                frame_queue.append(frame)
                t -= 1

            if self.detection_data[j][0] != prev_idx or j == 0:
                frame = Frame(self.detection_data[j][0])
            frame.add_box( Box(coordinates, self.detection_data[j][5]) )
            prev_idx = self.detection_data[j][0]
            j += 1

        self.last_added_frame_index = frame.frame_indx
        self.last_readed_row = j - 1

    def get_next_frame(self, frame_queue):
        """
        fill the buffer with the next frame and remove the first one.
        """

        i = self.last_readed_row
        if self.last_added_frame_index >= self.num_of_frames and len(frame_queue) >0:
            frame_queue.popleft()
            return
        curr = -1
        if i < len(self.detection_data):
            curr = self.detection_data[i][0]

        detection_exist = True
        '''print(self.last_added_frame_index)
        print(curr)'''
        if self.last_added_frame_index != curr:
            detection_exist = False
        frame = Frame(self.last_added_frame_index)
        image_path = join(self.detection_imgs_path, str(self.last_added_frame_index).zfill(8) + '.jpg')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame.set_img(img)
        if not detection_exist:
            self.last_added_frame_index += 1
            frame_queue.append(frame)
            return
        while i < len(self.detection_data) and self.detection_data[i][0] == curr:
            coordinates = (self.detection_data[i][1], self.detection_data[i][2],
                           self.detection_data[i][3], self.detection_data[i][4])
            box = Box(coordinates, self.detection_data[i][5])
            frame.add_box(box)
            i += 1

        for frame_box in frame.boxes:
            maxx = 0
            for prev_frame in frame_queue:

                for prev_box in prev_frame.boxes:
                    iou = box_box_iou(frame_box, prev_box)
                    if iou >= 0.4 and iou > maxx:
                        frame_box.ID = prev_box.ID
                        maxx = iou

        frame_queue.append(frame)
        self.last_added_frame_index = frame.frame_indx + 1
        self.last_readed_row = i

    def select_active_boxes(self, frame_queue, active_boxes, thresh, f):
        boxes = []
        if len(frame_queue) ==0:
            return  boxes
        box_freq = np.zeros(len(frame_queue[0].boxes))

        '''print("In Select active boxes")
        for f in frame_queue:
            for b in f.boxes:
                print("in frame {} box is {}".format(f.frame_indx, b))'''

        for current_box_indx in range(len(frame_queue[0].boxes)):
            current_box = frame_queue[0].boxes[current_box_indx]
            exist = False

            for active_box in active_boxes:
                if current_box.ID == active_box.ID:
                    #print("current box ID = {} is Found".format(current_box.ID))
                    exist = True
                    break

            if not exist:
                for active_box in active_boxes:
                    '''print(current_box)
                    print(active_box)
                    print(box_box_iou(current_box, active_box))'''
                    if box_box_iou(current_box, active_box) > thresh:

                        current_box.ID = active_box.ID
                        exist = True
                        break

            if exist:
                continue

            for i in range(1, len(frame_queue)):
                for box in frame_queue[i].boxes:
                    if(box_box_iou(current_box, box) >= thresh):
                        box_freq[current_box_indx] += 1
                        break
        #print(box_freq)
        for i in range(len(box_freq)):
            if(box_freq[i] >= 5):
                box = frame_queue[0].boxes[i]
                box.assign_ID(self.id_gen)
                #print("new assigned ID = {}".format(box.ID))
                boxes.append(box)
                for f in frame_queue:
                    for frame_box in f.boxes:
                        if box_box_iou(box, frame_box) >= thresh:
                            frame_box.ID = box.ID


        return boxes

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('/media/innolab/3448C50948C4CB36/tracking_test/detection_output.avi', fourcc, 60.0, (848, 480))
    detection_path = '/media/innolab/3448C50948C4CB36/tracking_test/detection_imgs'
    detection_csv_file = '/media/innolab/3448C50948C4CB36/tracking_test/detections.csv'
    fm = FrameManager(detection_csv_file, detection_path)
    queue = deque(maxlen=10)
    fm.fill_buffer(queue, 10)
    prev_frame = None
    while True:
        img = queue[0].img
        for box in queue[0].boxes:
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (255, 0, 0), 2)
        writer.write(img)
        if prev_frame is not None:
            a = []
            for current_box in queue[0].boxes:
                for prev_box in prev_frame.boxes:
                    a.append(current_box.box_iou(prev_box))
        prev_frame = queue[0]

        fm.get_next_frame(queue)