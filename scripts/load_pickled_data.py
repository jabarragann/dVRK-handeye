import numpy as np
import pickle
import cv2


# data_dict = np.load("data/data_00.npy")
# print(data_dict)


with open("data/handeye_data_v1/data_00.pkl", "rb") as file:
    data_dict = pickle.load(file)

print(data_dict)

print(data_dict["marker_rvec"])
print(data_dict["marker_rvec"])
print(data_dict["robot_jp"])
print(data_dict["robot_cp"])

window_name = "marker_detected"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)
cv2.imshow(window_name, data_dict["img"])

cv2.waitKey(0)
cv2.destroyAllWindows()
