from pathlib import Path
from bop_toolkit_lib.renderer_vispy import RendererVispy
from bop_toolkit_lib import misc
from bop_toolkit_lib.renderer_vispy import RendererVispy
from bop_toolkit_lib.visualization import draw_rect, write_text_on_image
import numpy as np
from dvrk_handeye.DataLoading import load_images_data, load_joint_data, load_poses_data
from dvrk_handeye.PSM_fk import compute_FK
from dvrk_handeye.opencv_utils import draw_axis
import cv2
from tqdm import tqdm
from scripts.hand_eye_calculation_methods.simple_validation2 import (
    draw_instrument_skeleton,
)

# fmt: off
mtx =   [1767.7722 ,    0.     ,  529.11477,
            0.     , 1774.33579,  510.58841,
            0.     ,    0.     ,    1.       ]
dist = [-0.337317, 0.500592, 0.001082, 0.002775, 0.000000]

cam_T_base = [[-0.722624032680848,    0.5730602138325281, -0.3865443036888354,  -0.06336454384831497], 
     [ 0.6870235345618597,   0.533741788848803,  -0.49307034568569336, -0.15304205999332426], 
     [-0.07624414961292797, -0.621869515379792,  -0.7794004974922103,   0.0664797333995826], 
     [0.0, 0.0, 0.0, 1.0]]

mtx = np.array(mtx).reshape(3, 3)
dist = np.array(dist)
cam_T_base = np.array(cam_T_base)
base_T_cam = np.linalg.inv(cam_T_base)

# fmt: on


def rendering_gt_single_obj(
    model_path: Path, intrinsic: np.ndarray, pose: np.ndarray, img: np.ndarray
) -> np.ndarray:

    intrinsic_mat = intrinsic
    extrinsic_mat = pose

    fx, fy, cx, cy = (
        intrinsic_mat[0, 0],
        intrinsic_mat[1, 1],
        intrinsic_mat[0, 2],
        intrinsic_mat[1, 2],
    )

    ren_rgb_info = np.zeros_like(img)

    width = 1280
    height = 1024
    renderer = RendererVispy(
        width, height, mode="rgb", shading="flat", bg_color=(0, 0, 0, 0)
    )

    # Load model
    model_color = [0.0, 0.8, 0.0]
    renderer.add_object(1, model_path, surf_color=model_color)
    ren_out = renderer.render_object(
        1, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], fx, fy, cx, cy
    )
    ren_out = ren_out["rgb"]

    obj_mask = np.sum(ren_out > 0, axis=2)
    ys, xs = obj_mask.nonzero()
    if len(ys):
        bbox_color = (0.5, 0.5, 0.5)
        text_color = (1.0, 1.0, 1.0)
        text_size = 16

        im_size = (obj_mask.shape[1], obj_mask.shape[0])
        bbox = misc.calc_2d_bbox(xs, ys, im_size)
        ren_rgb_info = draw_rect(ren_rgb_info, bbox, bbox_color)

        # text info
        text_loc = (bbox[0] + 2, bbox[1])
        txt_info = [dict(name="needle", val=0, fmt="")]
        ren_rgb_info = write_text_on_image(
            ren_rgb_info, txt_info, text_loc, color=text_color, size=text_size
        )

    # Combine with raw image
    vis_im_rgb = (
        0.5 * img.astype(np.float32)
        + 0.5 * ren_out.astype(np.float32)
        + 1.0 * ren_rgb_info.astype(np.float32)
    )

    vis_im_rgb[vis_im_rgb > 255] = 255
    vis_im_rgb = vis_im_rgb.astype(np.uint8)

    return vis_im_rgb, ren_out


def main():
    # raw images
    root_path = Path("datasets/20240213_212626_raw_dataset_handeye_raw_img_local")
    # rectified images
    # root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    model_path = Path("./temp/tool_pitch_link.ply")

    idx = 38
    img = load_images_data(root_path, idx)
    pose_data = load_poses_data(root_path)
    measured_jp = load_joint_data(root_path)
    base_T_wrist_pitch = compute_FK(measured_jp[idx], 5)

    pose = cam_T_base @ base_T_wrist_pitch

    pose[:3, 3] = pose[:3, 3] * 1000  # convert to mm

    vis_img_rgb, ren_out = rendering_gt_single_obj(model_path, mtx, pose, img)

    window_name = "Resized_Window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.imshow(window_name, vis_img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def multi_img_validation():
    from dvrk_handeye.opencv_utils import VideoWriter
    from dvrk_handeye.opencv_utils import VideoWriter

    # raw images
    root_path = Path("datasets/20240213_212626_raw_dataset_handeye_raw_img_local")

    # rectified images
    # root_path = Path("datasets/20240213_212744_raw_dataset_handeye_rect_img_local")

    model_path = Path("./temp/tool_pitch_link.ply")

    file_name = root_path.name + "_mesh_proj.mp4"
    video_writer = VideoWriter(
        output_dir="temp",
        width=1280,
        height=1024,
        fps=3,
        file_name=file_name,
    )

    measured_cp = load_poses_data(root_path)
    measured_jp = load_joint_data(root_path)

    with video_writer as writer:
        for i in tqdm(range(0, measured_cp.shape[0] - 20)):
            img = load_images_data(root_path, i)

            base_T_wrist_pitch = compute_FK(measured_jp[i], 5)
            pose = cam_T_base @ base_T_wrist_pitch
            pose[:3, 3] = pose[:3, 3] * 1000  # convert to mm
            vis_img_rgb, ren_out = rendering_gt_single_obj(model_path, mtx, pose, img)

            # img = draw_instrument_skeleton(
            #     img, measured_jp[i], measured_cp[i], i, debug=False
            # )
            writer.write_frame(i, vis_img_rgb)


if __name__ == "__main__":
    # main()
    multi_img_validation()
