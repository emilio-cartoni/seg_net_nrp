import torch
import numpy as np
import imageio
from src.model import PredNet

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import torchvision.transforms as transforms
import threading

import rospy

class PrednetSegmentation:
    def __init__(self, name, model_name, gpu_factor, device, class_names, n_frames):
        """Initialize ROS comm and Prednet PyTorch module """
        
        self.cv_bridge = CvBridge()

        self.frame_queue_lock = threading.Lock()
        self.frames = []

        self.device = device
        self.batch_size = batch_size
        self.n_channels = 3
        self.n_frames = n_frames

        self.n_classes = len(class_names)

        self.active_class_indices = (7, 9, 10)
        active_class_names = [ class_names[i] for i in self.active_class_indices ]

        torch.cuda.set_per_process_memory_fraction(gpu_factor, 0)
        self.model, _, _, _, _ = PredNet.load_model(model_name)
        self.model.eval()
        self.model.to(device)

        self.ros_seg_pubs = [ rospy.Publisher("/prednet_segmentation/{}/image".format(name), Image, queue_size=10) \
                                for name in active_class_names ]

        self.ros_frame_sub = rospy.Subscriber(cam_topic, Image, self._ros_frame_cb)

    def ExecuteStep(self, time):
        """Execute one segmentation step. """
        # Wait for first frame
        if not len(self.frames):
            print("No frames received")
            return
        
        # Segment images
        seg_image_sequence = []
        with torch.no_grad(), self.frame_queue_lock:
                t = 0
                for im in self.frames:
                    #image = image.to(device=device)
                    im = im[None, :, :, :]
                    image = im.to(device=device)
                    _, _, seg_image = self.model(image, t)
                    seg_image_numpy = seg_image.cpu().numpy()   # Shape of (1, n_classes, width, height)
                    seg_image_sequence.append(seg_image_numpy)
                    t += 1

        # Publish segmentation masks
        for i in range(0, len(self.active_class_indices)):
            ros_img = self.cv_bridge.cv2_to_imgmsg(seg_image_sequence[-1][0, self.active_class_indices[i], :, :], encoding="passthrough")
            self.ros_seg_pubs[i].publish(ros_img)

    def _convert_cv2_to_torch(self, cv_frame):
        """Convert from cv2 image to pytorch tensor"""
        image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB) 
        
        # Convert the image to PyTorch tensor 
        transform = transforms.ToTensor() 
        tensor = transform(image) 

        return tensor

    def _push_ros_frame(self, ros_frame):
        """Push ros frame onto self.frames"""
        # Convert to cv image
        cv_image = self.cv_bridge.imgmsg_to_cv2(ros_frame, desired_encoding="passthrough")
        tensor_frame = self._convert_cv2_to_torch(cv_image)

        # Insert into frame array, store self.n_frames previous frames
        with self.frame_queue_lock:
            if len(self.frames) < self.n_frames:
                self.frames.append(tensor_frame)
            else:
                for i in range(1, self.n_frames):
                    self.frames[i-1] = self.frames[i]
                
                self.frames[self.n_frames-1] = tensor_frame

    def _ros_frame_cb(self, data):
        # Push new frame onto stack
        self._push_ros_frame(data)

        self.ExecuteStep(0.0)


if __name__ == "__main__":
    try:
        rospy.init_node("prednet_segmentation")

        gpu_fact = rospy.get_param("segmentation_gpu_factor", 1.0)

        batch_size = 1
        im_w, im_h, n_channels = (300, 300, 3)
        n_frames = 30
        device = 'cuda'

        cam_topic="/camera/camera/image"

        seg_class_names = [
                'a_marbles',
                'apple',
                'banana',
                'adjustable_wrench',
                'flat_screwdriver',
                'mug',
                'phillips_screwdriver',
                'plate',
                'power_drill',
                'sugar_box',
                'tomato_soup_can'
        ]

        module = PrednetSegmentation("prednet_segmentation", 
                                     model_name="TA1_BU(64-128-256)_TD(64-128-256)_TL(H-H-H)_PL(0-)_SL(1-2)_DR(0-0-0)",
                                     gpu_factor=gpu_fact,
                                     device=device,
                                     class_names=seg_class_names,
                                     n_frames=n_frames)      
  
        #while not rospy.is_shutdown():
        #    module.RunOnce()
        #    time.sleep(1.0/120.0)
    
    except rospy.ROSInterruptException:
        pass

    rospy.spin()

