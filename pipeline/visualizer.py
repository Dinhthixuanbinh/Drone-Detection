import cv2
from loguru import logger

def draw_match(frame, similarity: float, frame_idx: int) -> None:
    cv2.putText(
        frame,
        f"Match: {similarity:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logger.info("Terminating visualization")
        cv2.destroyAllWindows()
