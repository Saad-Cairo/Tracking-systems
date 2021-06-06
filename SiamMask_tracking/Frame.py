import cv2
import matplotlib.pyplot as plt
class Frame:
    def __init__(self, frame_index, img=None):
        self.frame_indx = frame_index
        self.frame_name = self.extract_frame_name()
        self.boxes = []
        self.img = img
        self.polygons = []
        self.polygons_ID = []
        self.updated_boxes = []
        self.mask = []
        self.seg_coord = []
    def draw_img_with_boxes(self):
        for box in self.boxes:
            cv2.rectangle(self.img, (box.x1, box.y1), (box.x2, box.y2), (255, 0, 0), 2)
        plt.imshow(self.img)
        plt.show()
    def draw_img_with_polygons(self):
        # TODO
        pass
    def add_box(self, box):
        self.boxes.append(box)

    def set_img(self, img):
        self.img = img
    def add_polygon(self, polygon, polygon_ID):
        self.polygons.append(polygon)
        self.polygons_ID.append(polygon_ID)
    def extract_frame_name(self):
        return "F_"+str(self.frame_indx).zfill(9)+'.jpg'

    def add_updated_box(self, box):
        self.updated_boxes.append(box)

    def add_mask(self, mask, ad_id):
        self.mask.append([ad_id, mask])

    def get_coord_depend_seg(self, mask, x1, y1, x2, y2, ad_id):
        top_left_min = 3001
        top_left = (x1, y1)

        w = 0
        for i in range(int(y1), int(y2)):
            for j in range(int(x1), int(x2)):
                if self.validate_indices(i, j) and mask[i, j] > 0 and top_left_min > i + j:
                    top_left_min = i + j
                    top_left = (j, i)

        bottom_right_max = 0
        bottom_right = (x2, y2)

        for i in range(int(y2), int(y1), -1):
            for j in range(int(x2), int(x1), -1):
                if self.validate_indices(i, j) and mask[i, j] == True and bottom_right_max < i + j:
                    bottom_right_max = i + j
                    bottom_right = (j, i)

        self.seg_coord.append([ad_id, top_left[0], top_left[1], bottom_right[0], bottom_right[1]])

    def validate_indices(self, i, j):
        return (i >= 0 and i < self.img.shape[0]) and (j >= 0 and j < self.img.shape[1])
