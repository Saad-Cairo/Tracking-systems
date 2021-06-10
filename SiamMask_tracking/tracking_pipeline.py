from tracking_manager import *
from output_saver import *


class TrackingPipeline:
    def __init__(self, detection_csv_path, imgs_path, buffer_size, pickle_path, output_path, out_vid_path, make_vid,
                 sampling_ratio=60):
        self.detection_csv_path = detection_csv_path
        self.imgs_path = imgs_path
        self.output_coordinator = Output_coordinator(pickle_path, output_path, imgs_path)
        self.tracking_manager = TrackingManager(detection_csv_path, imgs_path, buffer_size,
                                                self.output_coordinator, out_vid_path, sampling_ratio)
        self.make_vid = make_vid

    def run_pipeline(self):
        tracks = self.tracking_manager.track()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        if self.make_vid:
            self.output_coordinator.make_output_video_from_pickle()

        return tracks


if __name__ == '__main__':
    # detection_path = '/media/innolab/3448C50948C4CB36/tested_videos_temp/test0/detection_images'
    detection_path = '/home/saad/Root/datasets/vot2017/helicopter-set'
    # detection_csv_file = '/media/innolab/3448C50948C4CB36/tested_videos_temp/test0/detections/detections.csv'
    detection_csv_file = '/home/saad/Root/datasets/tracking/helicopter_tracking.csv'

    pickle_path = '/home/saad/Root/datasets/tracking/tracking_output/tracking.pkl'
    output_path = '/home/saad/Root/datasets/tracking/tracking_output'
    out_vid_path = '/home/saad/Root/datasets/tracking/tracking_output/output_video.avi'
    # pickle_path = '/media/innolab/3448C50948C4CB36/tested_videos_temp/test0/detections/tracking.pkl'
    # output_path = '/media/innolab/3448C50948C4CB36/tested_videos_temp/test0/detections'
    # out_vid_path = '/media/innolab/3448C50948C4CB36/tested_videos_temp/test0/detections/output_video.avi'
    tracking_pipeline = TrackingPipeline(detection_csv_file, detection_path, 10,
                                         pickle_path, output_path, out_vid_path, make_vid=True)
    tracking_pipeline.run_pipeline()
