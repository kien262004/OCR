import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from skimage.filters import threshold_local
import imutils
from skimage import measure
import os
from torchvision import transforms
from PIL import Image
import torch
from utils.codeChar import config, label_to_char, label_to_int
from utils.tool import *
from utils.general import (
    non_max_suppression,
    xyxy2xywh,
)
from models.common import DetectMultiBackend
from models.recognite import CharaterRecognizer

conf_thres = 0.75
iou_thres = 0.75
classes = 10
agnostic_nms = True
max_det = 1000
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 2
transform = transforms.Compose([
  transforms.Resize((640,640)),
  transforms.ToTensor()
])



def parse_opt(known=False):
  parser = argparse.ArgumentParser()
  parser.add_argument("--dectection", type=str, default="pretrained/detection.pt", help="initial weights path for detection")
  parser.add_argument("--recognite", type=str, default="pretrained/recognite.pth", help="initial weights path for recognite")
  parser.add_argument("--imageDir", type=str, default="input", help="image dir path")
  parser.add_argument("--outputDir", type=str, default="output", help="output dir path")
  return parser.parse_known_args()[0] if known else parser.parse_args()



def run(dection, recognite, image_path, output_path):
  image = Image.open(image_path)
  trans_img = transform(image)
  trans_img = trans_img[:3].unsqueeze(0)

  res = detection(trans_img)
  pred = non_max_suppression(res, conf_thres, iou_thres)


  image = cv2.imread(image_path, cv2.IMREAD_COLOR)

  cropped_imgs = []
  result = []
  i = 0
  res_image = np.copy(image)
  # Duyệt qua các bounding boxes
  for det in pred:  # Duyệt qua từng ảnh trong batch
    if len(det):
      for *xyxy, conf, cls in det:
        x1, y1, x2, y2 = map(int, xyxy)  # Chuyển tọa độ sang integer

        x1 = int(x1*image.shape[1]/640)
        y1 = int(y1*image.shape[0]/640)
        x2 = int(x2*image.shape[1]/640)
        y2 = int(y2*image.shape[0]/640)

        cv2.rectangle(res_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)  # vẽ boundy box

        height = 100
        width = int(height * (x2-x1) / (y2-y1) * 1.2)

        # Cắt ảnh
        cropped_img = image[y1:y2, x1:x2]
        # cropped_img = cv2.resize(cropped_img, (100, 100))
        # cv2.imwrite(f'cropped_{x1}.png', cropped_img)
        cropped_V = segment_plate(cropped_img, width, height)
        # cv2.imwrite(f'cropped_V_{x1}.0.png', cropped_V)
        # create threshold for character
        T = threshold_local(cropped_V, 25, offset=5, method="gaussian")
        thresh = (cropped_V > T).astype("uint8") * 255
        t = thresh

        # invert bits for character is high bit
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400) # Now you can use imutils.resize
        thresh = cv2.medianBlur(thresh, 5)
        thresh[thresh>100] = 255
        thresh[thresh<=100] = 0
        # connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)

        candidates = segmentCharacter(labels, thresh)
        sto = []
        word = []
        candidates.sort(key = lambda x: (x[1][0], x[1][1]))
        before = candidates[0][1][0]
        i+=1
        for character, (y, x, h, w) in candidates:
          classifier_output = recognite(character)
          char = label_to_char[torch.argmax(classifier_output).item()]
          if (y - before) > 28 :
            sto.sort(key = lambda x: x[1])
            for i in range(len(sto)):
              word.append(sto[i][0])
            word.append('-')
            before = y
            sto.clear()
          sto.append((char, x))

        sto.sort(key = lambda x: x[1])
        for i in range(len(sto)):
          word.append(sto[i][0])
        result += [''.join(word)]
        label_position = (x1, y1 - 10)
        scale = image.shape[1] / 480
        cv2.putText(res_image, ''.join(word), label_position, font, font_scale * scale, (0, 255, 0), 2)
      cv2.imwrite(output_path, res_image)




def main(opt):

  detection_path, recognite_path, imageDir, outputDir = (
    opt.dectection,
    opt.recognite,
    opt.imageDir,
    opt.outputDir,
  )

  detection = DetectMultiBackend(detection_path)
  recognite = CharaterRecognizer(31, config)
  recognite.load_state_dict(torch.load(recognite_path))

  for image_name in os.listdir(imageDir):
    image_path = os.path.join(imageDir, image_name)
    output_path = os.path.join(outputDir, image_name)
    run(detection, recognite, image_path, output_path)


if __name__ == '__main__':
  opt = parse_opt()
  main(opt)
