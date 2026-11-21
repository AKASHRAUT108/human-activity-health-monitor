
import cv2

video_path = "data/raw/kth/videos/person01_boxing_d4_uncomp.avi"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())
print("Reported frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

print("\nDirect frame access test:")

for frame_number in [300, 304, 305, 306, 310, 320, 340, 350, 360, 369]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    print(
        f"Frame {frame_number}:",
        "DECODABLE" if success else "FAILED"
    )

cap.release()
