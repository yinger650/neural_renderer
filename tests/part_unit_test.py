import torch
import torch.nn as nn
import numpy as np
import neural_renderer as nr
import cv2
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, v_inp, v_tar, f, part_mask):
        super(Model, self).__init__()
        self.part_mask = part_mask
        self.renderer = nr.Renderer(camera_mode='look_at')
        self.renderer.eye = nr.get_points_from_angles(2, 0, 0)
        self.v = nn.Parameter(torch.from_numpy(v_inp.astype(np.float32)))
        self.register_buffer("f", torch.from_numpy(f.astype(np.int32)))
        self.register_buffer("v_tar", torch.from_numpy(v_tar.astype(np.float32)))
        silhouette_tar = self.renderer.render_silhouettes(self.v_tar.cuda(), self.f.cuda(), None,
                                                         part_mask=part_mask)

        self.target = silhouette_tar

    def forward(self):
        silhouette = self.renderer.render_silhouettes(self.v, self.f, None,
                                                         part_mask=self.part_mask)
        self.output = silhouette
        return self.output

    def concat_image(self, img):
        img_all = img.sum(axis=0)
        imgs = [img[i] for i in range(img.shape[0])]
        imgs.append(img_all)
        imgs_out = np.concatenate(imgs, axis=1)
        return imgs_out

    def show_now(self):
        imgs = self.output[0].detach().cpu().numpy()
        gt = self.target[0].detach().cpu().numpy()
        out_img = self.concat_image(imgs)
        out_gt = self.concat_image(gt)
        cv2.imshow("model output", np.concatenate([out_img, out_gt], 0))
        cv2.waitKey()


if __name__ == "__main__":
    faces = np.array([[[0, 2, 1], [3, 5, 4]]], dtype=np.int32)
    # vertices_begin = np.array([[[-0.5, 0, 1],
    #                             [0, 0, 1],
    #                             [-0.25, -0.5, 1],
    #                             [0, 0, 0.5],
    #                             [0.5, 0, 0.5],
    #                             [0.25, -0.5, 0.5]]])
    vertices_begin = np.array([[[-0.5, 0, 0],
                                [0, 0, 0],
                                [-0.25, -0.5, 0],
                                [-0.2, 0, 1],
                                [0.3, 0, 1],
                                [0.05, -0.5, 1]]])

    vertices_end = np.array([[[-0.5, 0, 0.5],
                              [0, 0, 0.5],
                              [-0.25, -0.5, 0.5],
                              [-0.2, 0, 0],
                              [0.5, 0, 0],
                              [0.05, -0.5, 0]]])
    print(faces.shape)
    print(vertices_begin.shape)
    model = Model(vertices_begin, vertices_end, faces, [0,1,2])
    model.cuda()
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    plt.axis([0, 1000, 0, 0.01])
    for i in range(1000):
        optimizer.zero_grad()
        output = model.forward()

        loss = criterion(output, model.target)
        loss.backward()
        print(model.v)
        print(model.v.grad)
        optimizer.step()
        print("Loss = {}".format(loss.detach().cpu().item()))
        model.show_now()
        # plt.scatter(i, loss.detach().cpu().item())
        # plt.pause(0.001)
    # plt.show()
