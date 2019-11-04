from __future__ import division
import math

import torch
import torch.nn as nn
import numpy

import neural_renderer as nr


class Renderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera_mode='projection',
                 K=None, R=None, t=None, P=None, bbox=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0,0,1],
                 near=0.1, far=100,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0]):
        super(Renderer, self).__init__()
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.cuda.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.cuda.FloatTensor(self.t)
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
            self.orig_size = orig_size
        elif self.camera_mode == 'projection_P':
            self.P = P
            if isinstance(self.P, numpy.ndarray):
                self.P = torch.cuda.FloatTensor(self.P)
            elif self.P is None or P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
                raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(P.shape[0], 1)
            self.orig_size = orig_size
        elif self.camera_mode == "projection_bbox":
            self.K = K
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if self.K is None or self.K.ndimension() != 3 or self.K.shape[1] != 3 or self.K.shape[2] != 3:
                raise ValueError('You need to provide a valid (batch_size)x3x3 intrinsic matrix')
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(K.shape[0], 1)
            self.bbox = bbox
            if self.bbox is None or bbox.ndimension() != 2 or bbox.shape[1] != 3:
                raise ValueError('You need to provide a valid (batch_size)x3 bbox info')


        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')


        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction 

        # rasterization
        self.rasterizer_eps = 1e-3

    def forward(self, vertices, faces, textures=None, mode=None, K=None, R=None, t=None, dist_coeffs=None, P=None, bbox=None, orig_size=None, part_mask=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        
        if mode is None:
            return self.render(vertices, faces, textures, K, R, t, dist_coeffs, P, bbox, orig_size, part_mask)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures, K, R, t, dist_coeffs, P, bbox, orig_size, part_mask)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces, K, R, t, dist_coeffs, P, bbox, orig_size, part_mask)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, K, R, t, dist_coeffs, P, bbox, orig_size, part_mask)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, P=None, bbox=None, orig_size=None, part_mask=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == "projection_P":
            if P is None:
                P = self.P
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection_P(vertices, P, dist_coeffs, orig_size)
        elif self.camera_mode == "projection_bbox":
            if K is None:
                K = self.K
            if bbox is None:
                bbox = self.bbox
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            vertices = nr.projection_bbox(vertices, K, dist_coeffs, bbox)


        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, part_mask, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, P=None, bbox=None, orig_size=None, part_mask=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == "projection_P":
            if P is None:
                P = self.P
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection_P(vertices, P, dist_coeffs, orig_size)
        elif self.camera_mode == "projection_bbox":
            if K is None:
                K = self.K
            if bbox is None:
                bbox = self.bbox
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            vertices = nr.projection_bbox(vertices, K, dist_coeffs, bbox)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, part_mask, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, P=None, bbox=None, orig_size=None, part_mask=None):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == "projection_P":
            if P is None:
                P = self.P
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection_P(vertices, self.P, self.dist_coeffs, self.orig_size)
        elif self.camera_mode == "projection_bbox":
            if K is None:
                K = self.K
            if bbox is None:
                bbox = self.bbox
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection_bbox(vertices, self.K, self.dist_coeffs, self.bbox)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, part_mask, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images

    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, P=None, bbox=None, orig_size=None, part_mask=None):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            if textures is not None:
                textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == "projection_P":
            if P is None:
                P = self.P
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection_P(vertices, self.P, self.dist_coeffs, self.orig_size)
        elif self.camera_mode == "projection_bbox":
            if K is None:
                K = self.K
            if bbox is None:
                bbox = self.bbox
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection_bbox(vertices, self.K, self.dist_coeffs, self.bbox)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        out = nr.rasterize_rgbad(
            faces, textures, part_mask, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth'], out['alpha']
