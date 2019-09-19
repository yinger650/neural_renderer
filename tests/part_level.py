import unittest
import os
import numpy as np
from skimage.io import imsave
import cv2
import torch
import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def ColorizeDepth(img_depth, near=1000, far=4000):
    median = np.median(img_depth)
    depth_norm = ((img_depth - near) / (far - near) * 255)
    depth_norm[depth_norm > 255] = 255
    depth_norm[depth_norm < 0] = 0
    depth_norm = depth_norm.astype(np.uint8)
    color_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return color_depth

class TestPartLevelRendering(unittest.TestCase):
    def test_case1(self):
        num_parts = 5
        renderer = nr.Renderer(camera_mode='look_at')

        vertices, faces, textures = nr.load_obj(
            os.path.join(data_dir, '1cde62b063e14777c9152a706245d48/model.obj'), load_texture=True)
        num_faces = faces.shape[0]
        part_mask = np.arange(0, num_faces, num_faces//num_parts)
        part_mask[-1] = num_faces
        renderer.eye = nr.get_points_from_angles(2, 15, 30)
        images, depth, silhouette = renderer.render(vertices[None, :, :], faces[None, :, :],
                                                    textures[None, :, :, :, :, :], part_mask=part_mask)
        images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
        silhouette = silhouette.detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()
        imsave(os.path.join(data_dir, 'car.png'), images[0])
        cv2.imshow("r", images[0, :, :, ::-1])
        cv2.imshow("d", ColorizeDepth(depth[0], 1.5, 2.5))
        for i in range(part_mask.shape[0]-1):
            sil = silhouette[0, i, :, :]
            cv2.imshow("s{}".format(i), sil)
            cv2.waitKey()
        cv2.waitKey()

    def part_backward(self):
        vertices = [
            [0.8, 0.8, 1.],
            [0.0, -0.5, 1.],
            [0.2, -0.4, 1.]]
        faces = [[0, 1, 2]]
        pass


if __name__ == '__main__':
    unittest.main()
