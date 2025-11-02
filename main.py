# main.py
from loguru import logger
from pipeline.config import Config
from pipeline.preprocessing import preprocess_image, get_video_frames
from pipeline.feature_extraction import extract_features
from pipeline.matcher import match_features
from pipeline.visualizer import draw_match

def main() -> None:
    logger.info("Initializing drone-based object detection pipeline")

    cfg = Config()

    # Load and preprocess target image
    target_img = preprocess_image(cfg.target_image_path, cfg.image_size)
    target_features = extract_features(target_img)

    # Open video stream and process frames
    for frame_idx, frame in get_video_frames(cfg.video_path, cfg.frame_skip):
        frame_pre = preprocess_image(frame, cfg.image_size)
        frame_features = extract_features(frame_pre)

        #Compute similarity
        similarity = match_features(target_features, frame_features)

        if similarity > cfg.match_threshold:
            logger.info(f"Match found at frame {frame_idx} | similarity={similarity:.3f}")
            draw_match(frame, similarity, frame_idx)
        else:
            logger.debug(f"No match in frame {frame_idx} | sim={similarity:.3f}")

    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
