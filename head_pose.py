import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scipy.spatial.transform import Rotation

# Landmark by index (68).
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image_width, image_height):
        """Init a pose estimator.

        Args:
            image_width (int): input image width
            image_height (int): input image height
        """
        self.size = (image_height, image_width)
        self.model_points_68 = self._get_full_model_points()
        
        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None


    def _get_full_model_points(self, filename='model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        # model_points[:, 0] *= -1
        model_points[:, 2] *= -1
        
        # # Personal edit. 
        # model_points[:, [1, 2]] = model_points[:, [2, 1]]
        # model_points[:, 2] *= -1

        return model_points


    def solve(self, points):
        """Solve pose with all the 68 image points
        Args:
            points (np.ndarray): points on image.

        Returns:
            Tuple: (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, points.astype('float32'), self.camera_matrix, self.dist_coeefs, flags=cv2.SOLVEPNP_ITERATIVE)
            # self.r_vec = rotation_vector
            # self.t_vec = translation_vector

        else: 
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68,
                points.astype('float32'),
                self.camera_matrix,
                self.dist_coeefs,
                rvec=self.r_vec,
                tvec=self.t_vec,
                useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)


    def euler_angles(self, pose):
        """
        Calculate the Euler angles (roll, pitch, yaw) for image filtering. 
        For facial images, 
            1) Roll: 
            2) Pitch: 
            3) Yaw: 
        Info on Euler angles: https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
        """
        # Get rotation & translation vector. 
        r_vec = pose[0]
        R = cv2.Rodrigues(r_vec)[0]            
        p_mat = np.hstack((R, np.array([[0], [0], [0]])))
        _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
        pitch, yaw, roll = u_angle.flatten()

        # I do not know why the roll axis seems flipted 180 degree. Manually by pass
        # this issue.
        if roll > 0:
            roll = 180-roll
        elif roll < 0:
            roll = -(180 + roll)
        
        return [-1 * pitch, roll, yaw]
    
    
    def visualize(self, image, pose, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        rotation_vector, translation_vector = pose
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)


    def draw_axes(self, img, pose):
        R, t = pose
        img = cv2.drawFrameAxes(img, self.camera_matrix,
                                self.dist_coeefs, R, t, 30)


    def show_3d_model(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter3D(x, y, z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        
        
def init_face_detector(): 
    
    # Initialize options.  
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        running_mode=VisionRunningMode.IMAGE,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)

    # Create MediaPipe detector. 
    detector = vision.FaceLandmarker.create_from_options(options)

    return detector     


def predict_head_pose(image, landmark_detector, pose_estimator): 
    
    # Landmark results. 
    results = landmark_detector.detect(image)
    face = results.face_landmarks[0]
    
    # Extract landmarks (global coords).
    landmarks = []
    if face is not None: 
        for idx in landmark_points_68: 
            x = int(face[idx].x * width)
            y = int(face[idx].y * height)
            # x = face[idx].x
            # y = face[idx].y
            landmarks.append([x,y])  # Add to list of landmarks. 
            cv2.circle(image.numpy_view(), (x, y), radius=1, color=(255, 0, 0), thickness=1)  # Draw for sanity check.

    # Estimate head pose. 
    landmarks = np.array(landmarks)  # 68 x 2 dim. 
    pose = pose_estimator.solve(landmarks)
    
    return pose



if __name__ == "__main__":
    
    # Load image.  
    # img_path = "/home/alex/바탕화면/Data/AffectNet/Images/Automatically_Annotated_Images/1/0c09149f8a157dadbe89e498d3450d27e30713f67a251b8d86903f6e.JPG"
    img_path = "/home/alex/바탕화면/AU/test_angle/roll.png"
    img = mp.Image.create_from_file(img_path)

    # Image dimensions. 
    height, width, _ = img.numpy_view().shape
    
    # Initialize landmark detector & pose estimator.  
    lm_detector = init_face_detector()
    pose_estimator = PoseEstimator(width, height)

    # Predict pose.  
    pose = predict_head_pose(img, lm_detector, pose_estimator)

    # Predict Euler angles. 
    euler_deg = pose_estimator.euler_angles(pose)
    print(f"Pitch (x rot): {euler_deg[0]}")
    print(f"Roll (y rot): {euler_deg[1]}")
    print(f"Yaw (z rot): {euler_deg[2]}")

    # pose_estimator.show_3d_model()
    
    img_np = img.numpy_view()
    
    pose_estimator.visualize(img_np, pose, color=(0, 255, 0))
    cv2.imshow("Preview", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)