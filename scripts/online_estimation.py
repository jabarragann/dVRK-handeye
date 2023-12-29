import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def show_rosbag_images(rosbag_file):
    bag = rosbag.Bag(rosbag_file, "r")
    bridge = CvBridge()

    for topic, msg, t in bag.read_messages(
        topics=["/jhu_daVinci/left/decklink/jhu_daVinci_left/image_raw"]
    ):
        if topic == "/jhu_daVinci/left/decklink/jhu_daVinci_left/image_raw":
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv2.imshow("ROS Image Viewer", cv_image)
                cv2.waitKey(20)  # Adjust the delay according to your frame rate
            except Exception as e:
                print(f"Error converting image: {e}")

    bag.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rosbag_file_path = "./data/rosbags/handeye_test.bag"
    show_rosbag_images(rosbag_file_path)
