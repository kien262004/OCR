import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
from skimage.filters import threshold_local
import imutils
from skimage import measure

def sort_points(points):
  # Sắp xếp 4 điểm theo thứ tự top-left, top-right, bottom-right, bottom-left
  # Đầu tiên, sắp xếp các điểm theo giá trị x (theo chiều ngang)
  sorted_points = sorted(points, key=lambda x: x[1])

  # Sau đó, sắp xếp theo y (theo chiều dọc)
  top_points = sorted(sorted_points[:2], key=lambda x: x[0])  # Top-left và Top-right
  bottom_points = sorted(sorted_points[2:], key=lambda x: x[0], reverse=True)  # Bottom-left và Bottom-right

  # Kết hợp các điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
  ordered_points = np.array(top_points + bottom_points, dtype="float32")
  return ordered_points

def euclidean_distance(point1, point2):
  return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_cornner(largest_contour):
  # Tạo một tập hợp điểm ngẫu nhiên đại diện cho vùng tứ giác
  # Xác định các góc tham chiếu
  contour_points = largest_contour.reshape(-1, 2)  # Định dạng lại thành mảng (n, 2)
  x_values = contour_points[:, 0]  # Tất cả giá trị x
  y_values = contour_points[:, 1]  # Tất cả giá trị y

  # Tìm x_min, x_max, y_min, y_max
  x_min = np.min(x_values)
  x_max = np.max(x_values)
  y_min = np.min(y_values)
  y_max = np.max(y_values)

  corners = (
      (x_min, y_min),
      (x_max, y_min),
      (x_max, y_max),
      (x_min, y_max),
  )

  # Tìm điểm xa nhất cho mỗi góc
  farthest_points = []
  for corner_point in corners:
      max_distance = 1e9
      farthest_point = None
      for point in contour_points:
          distance = euclidean_distance(corner_point, point)
          if distance < max_distance:
              max_distance = distance
              farthest_point = point
      farthest_points.append(farthest_point)

  box = np.array(farthest_points)
  return box

def segment_plate(image, width, height):
  # Chuyển ảnh thành mảng 2D (số pixel x 3)
  data = image.reshape((-1, 3))
  V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

  # Áp dụng K-means clustering
  kmeans = KMeans(n_clusters=3, random_state=42)  # Phân thành 5 cụm màu
  labels = kmeans.fit_predict(data)
  centroids = kmeans.cluster_centers_

  # Tìm cụm chiếm ưu thế (nhiều pixel nhất)
  unique, _ = np.unique(labels, return_counts=True)
  dominant_cluster = unique[np.argmax(np.average(centroids, axis=1))]  # Cụm chủ đạo

  # Tạo mặt nạ cho vùng màu chủ đạo
  mask = (labels == dominant_cluster).reshape(V.shape)
  mask = np.uint8(mask * 255)  # Chuyển sang dạng nhị phân (0 và 255)

  # Phân đoạn vùng màu chủ đạo
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = max(contours, key=cv2.contourArea)
  # rect = cv2.minAreaRect(largest_contour)
  # box = cv2.boxPoints(rect)  # Lấy 4 đỉnh của hình chữ nhật
  box = find_cornner(largest_contour)
  box = box.astype(int)
  ordered_points = sort_points(box)

  pts2 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])

  M = cv2.getPerspectiveTransform(ordered_points, pts2)

  # Áp dụng phép chiếu phối ảnh lên ảnh gốc
  result = cv2.warpPerspective(V, M, (width, height))
  return result

def segmentCharacter(labels, thresh):
  candidates = []
  for label in np.unique(labels):
    # if this is background label, ignore it
    if label == 0:
        continue

    # init mask to store the location of the character candidates
    mask = np.zeros(thresh.shape, dtype="uint8")
    mask[labels == label] = 255

    # find contours from mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)
        # rule to determine characters
        aspectRatio = w / float(h)
        solidity = cv2.contourArea(contour) / float(w * h)
        # Replace 'LpRegion' with 'thresh' to calculate heightRatio based on the current image
        heightRatio = h / float(thresh.shape[0])

        if 0.23 < aspectRatio < 1.0 and solidity > 0.25 and 0.33 < heightRatio < 1.0:
            # extract characters
            candidate = np.array(mask[y:y + h, x: x + w])
            #  square_candidate = convert2Square(candidate)
            square_candidate = cv2.resize(candidate, (28, 28), cv2.INTER_AREA)
            square_candidate = torch.tensor(square_candidate.reshape((28, 28, 1))) / 255.0

            square_candidate = square_candidate.permute(2, 0, 1)

            # pad_size = (28 - 20) // 2  # Padding đều ở cả 4 phía
            # padding = (pad_size, pad_size, pad_size, pad_size)  # (left, right, top, bottom)
            # # Áp dụng padding và mở rộng kích thước thành 28x28
            # square_candidate = F.pad(square_candidate, padding, mode='constant', value=0)

            square_candidate = square_candidate.unsqueeze(0)
            candidates.append((square_candidate, (y, x, h, w)))
  return candidates

