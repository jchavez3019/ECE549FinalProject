from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

USE_CUDA = False

if (USE_CUDA):
    device = 'cuda:0'
else:
    device = 'cpu'


def returnDetectedFaces(img_path):

    # Create face detector
    mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device=device)

    # load in the image
    frame = Image.open(img_path).convert("RGB")

    # plt.figure(figsize=(20,20))
    # plt.imshow(frame)
    # plt.axis('off')
    # plt.show()

    # Detect face
    faces = mtcnn(frame)

    # convert faces into a PIL image array
    pil_images = []
    for face in faces:
        pil_images.append(Image.fromarray(np.uint8(face.permute(1,2,0).int().numpy())))

    return pil_images