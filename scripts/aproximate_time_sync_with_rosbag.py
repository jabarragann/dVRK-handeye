import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import message_filters
import tf_conversions.posemath as pm

# fmt: off
mtx = np.array( [ [1747.03274843401, 0.0, 521.0492603841079], [0.0, 1747.8694061586648, 494.32395180615987], [0.0, 0.0, 1.0], ])
dist = np.array( [ -0.33847195608374453, 0.16968704500434714, 0.0007293228134352138, 0.005422675750927001, 0.9537762252401928, ])
marker_size = 0.01

Y = np.array([[ 0.15755188, 0.51678588, 0.84149258, 0.26527036],
              [-0.89806276,-0.27940152, 0.33973234,-0.02360361],
              [ 0.41068319,-0.80923862, 0.42008592,-0.08475506],
              [ 0.        , 0.        , 0.        , 1.        ]])
Y_inv = np.linalg.inv(Y)
# fmt: on

robot_kin = np.eye(4)


def callback(*msgs):
    # print("msg received")
    # print(msgs[1].pose.position)

    global robot_kin
    robot_kin = pm.toMatrix(pm.fromMsg(msgs[1].pose))


def show_rosbag_images(rosbag_file):
    global robot_kin
    bag = rosbag.Bag(rosbag_file, "r")
    bridge = CvBridge()

    window_name = "ROS Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    topic_dict = dict(
        img_topic="/jhu_daVinci/left/decklink/jhu_daVinci_left/image_raw",
        kin_topic="/PSM2/measured_cp",
    )
    topic_list = list(topic_dict.values())

    img_subs = message_filters.Subscriber(topic_dict["img_topic"], Image)
    kin_subs = message_filters.Subscriber(topic_dict["kin_topic"], PoseStamped)
    sub_list = [img_subs, kin_subs]
    sub_dict = dict([(s.topic, s) for s in sub_list])

    ts = message_filters.ApproximateTimeSynchronizer(
        sub_list, queue_size=20, slop=0.1, allow_headerless=False
    )
    ts.registerCallback(callback)

    for topic, msg, t in bag.read_messages(topics=topic_list):
        if topic == "/jhu_daVinci/left/decklink/jhu_daVinci_left/image_raw":
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                pose = Y_inv @ robot_kin
                tvec = pose[:3, 3]
                rvec = cv2.Rodrigues(pose[:3, :3])[0]

                points_3d = np.array([[[0, 0, 0]]], np.float32)
                points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)
                print(points_2d)

                cv_image = cv2.drawFrameAxes(cv_image, mtx, dist, rvec, tvec, 0.01)
                cv2.imshow(window_name, cv_image)
                key = cv2.waitKey(20)  # Adjust the delay according to your frame rate

                if key == ord("q") or key == 27:
                    break

            except Exception as e:
                print(f"Error converting image: {e}")

        subscriber = sub_dict.get(topic)
        if subscriber:
            # # Show some output to show we are alive
            # if message_idx % 1000 == 0:
            #     print(
            #         "Message #{}, Topic: {}, message stamp: {}".format(
            #             message_idx, topic, msg.header.stamp
            #         )
            #     )
            subscriber.signalMessage(msg)

    bag.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("rosbag sync")
    rosbag_file_path = "./data/rosbags/handeye_test.bag"
    show_rosbag_images(rosbag_file_path)
