


class Box:
    def __init__(self, coordinates, type, ID=-1):
        self.x1, self.y1, self.x2, self.y2 = coordinates
        self.ID = ID
        self.type = type
        self.w, self.h = self.covert_to_ROI()
    def __str__(self):
        return f"ID = {self.ID}, x1 = {self.x1}, y1 = {self.y1}, x2 = {self.x2}, y2 = {self.y2}, type = {self.type}"

    def box_iou(self, box):
        '''
        calculate intersection over union between 2 boxes
        :param box1: Box object
        :param box2: Box object
        :return: IOU between the 2 boxes.
        '''

        xA = max(self.x1, box.x1)
        xB = min(self.x2, box.x2)

        yA = max(self.y1, box.y1)
        yB = min(self.y2, box.y2)

        inter = abs(max(0, xB - xA) * max(0, yB - yA))

        box1Area = (self.x2 - self.x1) * (self.y2 - self.y1)
        box2Area = (box.x2 - box.x1) * (box.y2 - box.y1)

        union = box1Area + box2Area - inter
        return inter / union

    def covert_to_ROI(self):
        return self.x2 - self.x1 , self.y2 - self.y1

    def assign_ID(self, id_gen):
        self.ID = id_gen.assign_ID()


