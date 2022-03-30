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
    def __init__(self, name, gpu_factor, device, batch_size, n_channels, im_w, im_h, n_frames, class_names, cam_topic):
        """Initialize ROS comm and Prednet PyTorch module """
        #super().__init__(name)
        
        self.cv_bridge = CvBridge()

        self.frame_queue_lock = threading.Lock()
        self.frames = []

        self.device = device
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_frames = n_frames

        torch.cuda.set_per_process_memory_fraction(gpu_factor, 0)
        self.model = PredNet(model_name='my_model',
                            n_classes=len(class_names),
                            n_layers=3,
                            seg_layers=(1, 2),
                            bu_channels=(64, 128, 256),
                            td_channels=(64, 128, 256),
                            do_segmentation=True,
                            device=device)
        
        self.model.eval()

        self.ros_seg_pubs = [ rospy.Publisher("/prednet_segmentation/{}/image".format(name), Image, queue_size=10) \
                                for name in class_names ]

        self.ros_frame_sub = rospy.Subscriber(cam_topic, Image, self._ros_frame_cb)

    def ExecuteStep(self, time):
        """Execute one segmentation step. """
        # Wait for first frame
        if not len(self.frames):
            #res = ModuleExecutionResult()
            #res.PauseTime = rospy.Time(0.0)
            #res.ExecutionTime = rospy.Time(0.01)
            #return res
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
                    seg_image_numpy = seg_image.cpu().numpy()   # Shape of (1, n_channels, width, height)
                    seg_image_sequence.append(seg_image_numpy)
                    t += 1

        # Publish segmentation masks
        for i in range(self.n_channels):
            ros_img = self.cv_bridge.cv2_to_imgmsg(seg_image_sequence[-1][0,i,:,:], encoding="passthrough")
            self.ros_seg_pubs[i].publish(ros_img)


        #res = ModuleExecutionResult()
        #res.PauseTime = rospy.Time(0.1)
        #res.ExecutionTime = rospy.Time(1.0)
        #return res

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
        #print("Adding frame")
        self._push_ros_frame(data)

        self.ExecuteStep(0.0)


if __name__ == "__main__":
    try:
        rospy.init_node("prednet_segmentation")

        gpu_fact = rospy.get_param("segmentation_gpu_factor", 1.0)

        batch_size = 1
        n_channels = 3
        im_w, im_h, n_channels = (300, 300, 3)
        n_frames = 30
        device = 'cuda'

        cam_topic="/camera/camera/image"

        seg_class_names = [
            "hammer",
            "cube",
            "cheez_its"
        ]

        module = PrednetSegmentation("prednet_segmentation",
                                     gpu_factor=gpu_fact,
                                     device=device,
                                     batch_size=batch_size, 
                                     n_channels=n_channels, 
                                     im_w=im_w, im_h=im_h, 
                                     n_frames=n_frames,
                                     class_names=seg_class_names,
                                     cam_topic=cam_topic)
        
        #while not rospy.is_shutdown():
        #    module.RunOnce()
        #    time.sleep(1.0/120.0)
    
    except rospy.ROSInterruptException:
        pass

    rospy.spin()

