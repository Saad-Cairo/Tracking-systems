import pickle
from box import *
from tqdm import tqdm
import cv2
from os.path import join
from matplotlib import pyplot as plt
from tracks import *

class Output_coordinator:

    def __init__(self, pickle_path, output_path, detection_imgs_path):
        self.general_dictionary = {}
        self.output_path = output_path
        self.pickle_path = pickle_path
        self.detection_images_path = detection_imgs_path
        self.connected_tracks = {}
    def add_frame(self, frame):

        for output_box in frame.updated_boxes:
            '''seg_coords = np.array(frame.seg_coord)
            seg_coords = seg_coords[seg_coords[:, 0] == output_box.ID]
            seg_coords = seg_coords[0]'''

            track_list = [frame.frame_name]
            track_list.extend([output_box.x1, output_box.y1, output_box.x2, output_box.y2])
            track_list.append(output_box.type)
            if "AD{}".format(output_box.ID) not in self.general_dictionary:
                self.general_dictionary["AD{}".format(output_box.ID)] = []

            self.general_dictionary["AD{}".format(output_box.ID)].append(track_list)

            #self.general_dictionary["AD{}".format(ID)].append()

    def connect_tracks(self):


        connect_tracks_from_pickle(self.pickle_path, self.output_path, min_time_window=120, min_dist_window=120)

        file_path = self.output_path + '/speed_linked_tracks.pkl'
        file = open(file_path, 'rb')
        self.connected_tracks = pickle.load(file)
        file.close()

    def save_results_in_pickle(self):
        file = open(self.pickle_path, 'wb')
        pickle.dump(self.general_dictionary, file, pickle.HIGHEST_PROTOCOL)
        file.close()
        self.connect_tracks()
        return self.connected_tracks


    def make_output_video_from_pickle(self):
        dictt = {}
        self.connect_tracks()
        #print(self.connected_tracks)
        for key, value in tqdm(self.connected_tracks.items()):
            for element in value:
                if element[0] not in dictt:
                    dictt[element[0]] = []
                dictt[element[0]].append([key, element[1], element[2], element[3], element[4], element[5]])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(join(self.output_path, 'frag_output.avi'), fourcc, 60.0, (848, 480))


        for key, value in sorted(dictt.items()):
            # print(self.detection_images_path)
            # print(key)
            key2 = key[3:]
            image_path = join(self.detection_images_path, key2)
            # print(image_path)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for box in value:
                cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (255, 0, 0), 2)

                #cv2.circle(img, (box[5], box[6]), 2, (255, 0, 0), 2)
                #cv2.circle(img, (box[7], box[8]), 2, (255, 0, 0), 2)
                w = box[3] - box[1]
                h = box[4] - box[2]
                center = (int(w/2 + box[1]) , int(h/2 + box[2]))
                cv2.putText(img, str(box[0][2:]), center,  cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            img = img [:, :, [2, 1, 0]]

            video_writer.write(img)
            print("frame has been written")