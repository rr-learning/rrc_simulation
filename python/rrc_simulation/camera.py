import pybullet
from scipy.spatial.transform import Rotation


class Camera(object):
    """Represents a camera in the simulation environment."""

    def __init__(
        self,
        camera_position,
        camera_orientation,
        image_size=(720, 540),
        pybullet_client=pybullet,
        **kwargs,
    ):
        """Initialize.

        Args:
            camera_position:  Position (x, y, z) of the camera w.r.t. the world
                frame.
            camera_orientation:  Quaternion (x, y, z, w) representing the
                orientation of the camera.
            image_size:  Tuple (width, height) specifying the size of the
                image.
            pybullet_client:  Client for accessing the simulation.  By default
                the "pybullet" module is used directly.
        """

        self._kwargs = kwargs

        self._pybullet_client = pybullet_client
        self._width = image_size[0]
        self._height = image_size[1]

        camera_rot = Rotation.from_quat(camera_orientation)
        target_position = camera_rot.apply([0, 0, 1])
        camera_up_vector = camera_rot.apply([0, -1, 0])

        self._view_matrix = self._pybullet_client.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=camera_up_vector,
            **self._kwargs,
        )

        self._proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=52,
            aspect=float(self._width) / self._height,
            nearVal=0.001,
            farVal=100.0,
            **self._kwargs,
        )

    def get_image(self):
        """Get a rendered image from the camera.

        Returns:
            (array, shape=(height, width, 3)):  Rendered RGB image from the
                simulated camera.
        """
        # FIXME: images are upside down
        (_, _, img, _, _) = self._pybullet_client.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            **self._kwargs,
        )
        # remove the alpha channel
        return img[:, :, :3]


class TriFingerCameras:
    """Simulate the three cameras of the TriFinger platform."""

    def __init__(self, **kwargs):
        self.cameras = [
            # camera60
            Camera(
                camera_position=[0.2496, 0.2458, 0.4190],
                camera_orientation=[0.3760, 0.8690, -0.2918, -0.1354],
                **kwargs,
            ),
            # camera180
            Camera(
                camera_position=[0.0047, -0.2834, 0.4558],
                camera_orientation=[0.9655, -0.0098, -0.0065, -0.2603],
                **kwargs,
            ),
            # camera300
            Camera(
                camera_position=[-0.2470, 0.2513, 0.3943],
                camera_orientation=[-0.3633, 0.8686, -0.3141, 0.1220],
                **kwargs,
            ),
        ]

    def get_images(self):
        """Get images.

        Returns:
            List of images, one per camera.  Order is [camera60, camera180,
            camera300].  See Camera.get_image() for details.
        """
        return [c.get_image() for c in self.cameras]
