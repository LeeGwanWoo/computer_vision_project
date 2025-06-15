import cv2
import pytesseract
import os
import pickle

from utils import load_video, write_video
from detection import extract_subtitle_pixel_from_video
from inpainting_no_blending import apply_inpainting

DETECTIONS_FILE = "./cache/detections.pkl"

if __name__ == "__main__":
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  # 1) 파라미터 설정
  video_path = './videos/Squid_game - Trim.mp4'
  crop_range = {"top": 870, "bottom": 1000, "left": 120, "right": 1800}
  target_color = [255, 255, 255]
  pixel_highlighting_tolerances = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] 
  conf_thresholds = [0, 10, 50]
  psm_modes = {
      "psm6": '--oem 3 --psm 6 -l kor+eng',
      "psm8": '--oem 3 --psm 8 -l kor+eng',
      "psm10": '--oem 3 --psm 10 -l kor+eng'
  }
  # Inpaint 파라미터
  level = 2               
  num_passes = 4           
  radius_patch = 15        
  kernel_size = 9          
  feather_sigma = 1.5      
  threshold_schedule = [10, 8, 5, 3]  

  fine_pass = True
  fine_radius = 5
  fine_kernel = 5
  fine_feather = 0.5
  fine_threshold = 3

  # 2) 프레임 로드
  frames = load_video(video_path)

  # 3) 자막 검출 → mask & box 리스트
  if os.path.exists(DETECTIONS_FILE):
      print("검출 결과 로드 중…")
      with open(DETECTIONS_FILE, "rb") as f:
          boxes, masks = pickle.load(f)
  else:
      print("자막 검출 시작…")
      boxes, masks = extract_subtitle_pixel_from_video(
          video_path,
          target_color,
          crop_range,
          pixel_highlighting_tolerances,
          psm_modes,
          conf_thresholds
      )
      os.makedirs(os.path.dirname(DETECTIONS_FILE), exist_ok=True)
      with open(DETECTIONS_FILE, "wb") as f:
          pickle.dump((boxes, masks), f)
      print(f"검출 결과 저장: {DETECTIONS_FILE}")

  # 4) Inpainting 적용
  inpainted = apply_inpainting(
      frames, masks, boxes,
      level, num_passes, radius_patch,
      kernel_size, feather_sigma, threshold_schedule,
      fine_pass, fine_radius, fine_kernel, fine_feather, fine_threshold
  )

  # 5) 비디오 쓰기
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  cap.release()
  
  os.makedirs('./results', exist_ok=True)
  write_video(inpainted, './results/inpainted_video.mp4', fps, (width, height))
  print("Inpainted video saved to ./results/inpainted_video.mp4")

    
    
    