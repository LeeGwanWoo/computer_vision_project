import cv2
import matplotlib.pyplot as plt

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def write_video(frames, out_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    for f in frames:
        writer.write(f)
    writer.release()

def crop_image(image, crop_range):
    top = crop_range["top"]
    bottom = crop_range["bottom"]
    left = crop_range["left"]
    right = crop_range["right"]
    
    return image[top:bottom, left:right]

def crop_video(frames, crop_range):
    top = crop_range["top"]
    bottom = crop_range["bottom"]
    left = crop_range["left"]
    right = crop_range["right"]
    
    cropped_frames = []
    for frame in frames:
        if frame is not None:
            cropped_frame = frame[top:bottom, left:right]
            cropped_frames.append(cropped_frame)
    return cropped_frames

def display_frame(frame, title="Frame"):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_cropped_frame(frame, crop_range, title="Cropped Frame Comparison"):
    plt.figure(figsize=(12, 5)) # 전체 그림 크기 유지 또는 조절
    plt.suptitle(title) # 전체 그림의 제목

    # 1. 원본 이미지에 크롭 영역 표시
    original_frame_with_crop_rect = frame.copy()
    cv2.rectangle(original_frame_with_crop_rect, 
                    (crop_range["left"], crop_range["top"]), 
                    (crop_range["right"], crop_range["bottom"]), 
                    (0, 255, 0), 2) # 초록색 사각형
    plt.subplot(1, 2, 1) # 1행 2열 중 첫 번째
    plt.imshow(cv2.cvtColor(original_frame_with_crop_rect, cv2.COLOR_BGR2RGB))
    plt.title("Original with Crop Area")
    plt.axis('off')

    # 2. 실제 크롭된 이미지 표시
    cropped_img = crop_image(frame, crop_range) # crop_image 함수 사용
    plt.subplot(1, 2, 2) # 1행 2열 중 두 번째
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image")
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # suptitle과의 간격 조절
    plt.show()
  
if __name__ == "__main__":
    crop_range = {"top": 870, "bottom": 1000, "left": 120, "right": 1800}
    video_path = './videos/Squid_game.mp4'
    sample_index = 1500

    frames = load_video(video_path)
    cropped_images = crop_video(frames, crop_range)
    image_shape = frames[sample_index].shape
    sample_original_image = frames[sample_index]
    sample_cropped_image = cropped_images[sample_index]
    display_cropped_frame(sample_original_image, crop_range)