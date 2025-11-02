from dataclasses import dataclass

@dataclass
class Config:
    target_image_path: str = "data/target.jpg"
    video_path: str = "data/drone_video.mp4"
    image_size: tuple[int, int] = (224, 224)
    match_threshold: float = 0.75
    frame_skip: int = 5  
