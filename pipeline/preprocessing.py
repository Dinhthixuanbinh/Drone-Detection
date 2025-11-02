import cv2
import numpy as np
from loguru import logger
from typing import Generator, Tuple

def preprocess_image(image_input: str | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            logger.error(f"Cannot load image: {image_input}")
            raise FileNotFoundError(image_input)
    else:
        img = image_input

    img_resized = cv2.resize(img, size)
    img_norm = img_resized / 255.0
    return img_norm.astype("float32")

def get_video_frames(video_path: str, frame_skip: int) -> Generator[Tuple[int, np.ndarray], None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        raise FileNotFoundError(video_path)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            yield idx, frame
        idx += 1

    cap.release()
