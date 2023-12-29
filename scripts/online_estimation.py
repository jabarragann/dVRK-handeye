import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


# fmt: off
mtx = np.array( [ [1747.03274843401, 0.0, 521.0492603841079], [0.0, 1747.8694061586648, 494.32395180615987], [0.0, 0.0, 1.0], ])
dist = np.array( [ -0.33847195608374453, 0.16968704500434714, 0.0007293228134352138, 0.005422675750927001, 0.9537762252401928, ])
marker_size = 0.01
# fmt: on


def show_rosbag_images(rosbag_file):
    bag = rosbag.Bag(rosbag_file, "r")
    bridge = CvBridge()

    window_name = "ROS Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    for topic, msg, t in bag.read_messages(
        topics=["/jhu_daVinci/left/decklink/jhu_daVinci_left/image_raw"]
    ):
        if topic == "/jhu_daVinci/left/decklink/jhu_daVinci_left/image_raw":
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv2.imshow(window_name, cv_image)
                key = cv2.waitKey(20)  # Adjust the delay according to your frame rate

                if key == ord("q") or key == 27:
                    break

            except Exception as e:
                print(f"Error converting image: {e}")

    bag.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rosbag_file_path = "./data/rosbags/handeye_test.bag"
    show_rosbag_images(rosbag_file_path)
