import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
from utils import *

# target cololr extraction and highlighting functions
def get_target_color(cropped_image, target_pixel):
    y, x = target_pixel
    target_color = cropped_image[y, x].astype(int)
    print(f"Target Pixel: {target_pixel}, Color (RGB): {target_color}")

    # 시각화: 선택한 픽셀 위치에 점 표시
    plt.figure(figsize=(5, 5))
    plt.imshow(cropped_image)
    plt.scatter([x], [y], edgecolors='red', facecolors='none', s=30, marker='o', label='Target Pixel')
    plt.title("Cropped Image with Target Pixel")
    plt.legend()
    plt.axis('off')
    plt.show()

    return target_color

def highlight_subtitle_area(cropped_image, target_color, pixel_color_tolerance):
    lower = np.array([max(0, c - pixel_color_tolerance) for c in target_color])
    upper = np.array([min(255, c + pixel_color_tolerance) for c in target_color])

    cropped_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    lower_bgr = np.array([lower[2], lower[1], lower[0]])
    upper_bgr = np.array([upper[2], upper[1], upper[0]])
    mask = cv2.inRange(cropped_bgr, lower_bgr, upper_bgr)

    highlighted_image = np.full_like(cropped_image, 0)
    highlighted_image[mask == 255] = [255, 255, 255]

    highlight_result = {
        "cropped_image": cropped_image.copy(),
        "highlighted_image": highlighted_image.copy(),
        "pixel_color_tolerance": pixel_color_tolerance,
        "mask": mask.copy(),
        "white_pixel_count": np.sum(mask == 255),
    }
    return highlight_result

def highlight_subtitle_area_with_multi_tolerances(cropped_image, target_color, tolerances):
    results = []
    for tolerance in tolerances:
        result = highlight_subtitle_area(cropped_image, target_color, tolerance)
        results.append(result)
    return results

def visualize_highlight_result(image, target_color, highlighted_image_result):
    print(f"Target Color (RGB): {target_color}")

    # 원본 크롭 이미지 + 점 표시
    plt.figure(figsize=(12, 5)) # figsize 조절 가능
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Cropped (with Selected Point)")
    plt.axis('off')
    # plt.legend(loc='upper right') # 원본 이미지에는 특별한 legend가 없을 수 있으므로 주석 처리 또는 수정
    
    # 단일 하이라이트된 이미지 시각화
    plt.subplot(1, 2, 2)
    plt.imshow(highlighted_image_result["highlighted_image"])
    plt.title(f"Highlighted - pixel_color_Tolerance = {highlighted_image_result['pixel_color_tolerance']}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
# box extraction
def find_boundingbox_from_highlighted_image(highlighted_image, tolerance, psm_mode, conf_threshold):

    gray = cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.01)
    _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    image_height, image_width = dilated.shape

    ocr_result = pytesseract.image_to_data(
        dilated, output_type=Output.DICT, config=psm_mode
    )
    n_boxes = len(ocr_result['level'])
    boxes = []

    for i in range(n_boxes):
        text = ocr_result['text'][i].strip()
        conf = int(ocr_result['conf'][i])
        x = ocr_result['left'][i]
        y = ocr_result['top'][i]
        w = ocr_result['width'][i]
        h = ocr_result['height'][i]

        if (conf > conf_threshold and text and
            x > 0 and y > 0 and x + w < image_width and y + h < image_height):

            boxes.append({
                "tolerance": tolerance,
                "psm_mode": psm_mode,
                "conf_threshold": conf_threshold,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
                "text": text
            })
    return boxes

def find_boundingbox_from_highlighted_results(highlight_results, psm_modes, conf_thresholds):
    all_boxes = []
    for result in highlight_results:
        highlighted_image = result['highlighted_image']
        pixel_color_tolerance = result['pixel_color_tolerance']
        
        for psm_mode in psm_modes:
            for conf_threshold in conf_thresholds:
                boxes = find_boundingbox_from_highlighted_image(
                    highlighted_image, pixel_color_tolerance, psm_modes[psm_mode], conf_threshold
                )
                if boxes:
                    all_boxes.extend(boxes)
    return all_boxes

def visualize_boxes(image, list_of_boxes):
    print(f"총 {len(list_of_boxes)}개의 박스가 발견되었습니다.")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for box_info in list_of_boxes: # 리스트의 각 박스 정보에 대해 반복
        x = box_info['x']
        y = box_info['y']
        w = box_info['w']
        h = box_info['h']
        text = box_info.get('text', '') # text 키가 없을 경우 빈 문자열
        conf = box_info.get('conf', -1) # conf 키가 없을 경우 -1

        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()
    
# box preprocessing
def color_similar(pixel, target_color, tolerance=30):
    pixel = np.array(pixel)
    target_color = np.array(target_color)
    return np.all(np.abs(pixel.astype(int) - target_color.astype(int)) <= tolerance)

def box_trimming(image, box, target_color, box_trimming_tolerance=30):
    x0, y0, w, h = box["x"], box["y"], box["w"], box["h"]
    x_min, y_min, x_max, y_max = x0, y0, x0 + w, y0 + h

    # 이미지 경계에 닿은 박스는 제외
    if x_min == 0 or y_min == 0 or x_max == image.shape[1] or y_max == image.shape[0]:
        return None

    # 위쪽 줄이기
    for yy in range(y_min, y_max):
        line = image[yy, x_min:x_max]
        if np.any([color_similar(px, target_color, box_trimming_tolerance) for px in line]):
            y_min = yy
            break

    # 아래쪽 줄이기
    for yy in range(y_max-1, y_min-1, -1):
        line = image[yy, x_min:x_max]
        if np.any([color_similar(px, target_color, box_trimming_tolerance) for px in line]):
            y_max = yy + 1
            break

    # 왼쪽 줄이기
    for xx in range(x_min, x_max):
        col = image[y_min:y_max, xx]
        if np.any([color_similar(px, target_color, box_trimming_tolerance) for px in col]):
            x_min = xx
            break

    # 오른쪽 줄이기
    for xx in range(x_max-1, x_min-1, -1):
        col = image[y_min:y_max, xx]
        if np.any([color_similar(px, target_color, box_trimming_tolerance) for px in col]):
            x_max = xx + 1
            break

    # 최소 크기 보장
    if x_max > x_min and y_max > y_min:
        trimmed_box = box.copy()
        trimmed_box["x"] = x_min
        trimmed_box["y"] = y_min
        trimmed_box["w"] = x_max - x_min
        trimmed_box["h"] = y_max - y_min
    
    return trimmed_box

def boxes_trimming(image, all_boxes, target_color, box_trimming_tolerance=30):
    trimmed_boxes = []
    for box in all_boxes:
        trimmed_box = box_trimming(image, box, target_color, box_trimming_tolerance)
        if trimmed_box is not None:
            trimmed_boxes.append(trimmed_box)
    return trimmed_boxes
  
# histogram and mode extraction
def histogram_ylength(boxes):
    y_lengths = [box['h'] for box in boxes]
    bin_width = 5
    min_val = np.min(y_lengths)
    max_val = np.max(y_lengths)
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    plt.figure(figsize=(10, 5))
    plt.hist(y_lengths, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Y Lengths of Bounding Boxes')
    plt.xlabel('Y Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def find_ylength_mode_boxes(boxes):
    y_lengths = [box['h'] for box in boxes]
    bin_width = 5
    min_val = np.min(y_lengths)
    max_val = np.max(y_lengths)
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # 최빈 bin 찾기
    hist, bin_edges = np.histogram(y_lengths, bins=bins)
    mode_index = np.argmax(hist)

    # 최빈 bin에 속하는 box만 추출
    mode_bin = (bin_edges[mode_index], bin_edges[mode_index + 1])
    mode_boxes = [box for box in boxes if mode_bin[0] <= box['h'] < mode_bin[1]]
    print(f"최빈 y 길이 범위: {mode_bin}, 개수: {len(mode_boxes)}")
    return mode_boxes

# subtitle mask and bounding box extraction
def get_final_bounding_box(boxes, crop_area):
    x_min = min(box['x'] for box in boxes)
    y_min = min(box['y'] for box in boxes)
    x_max = max(box['x'] + box['w'] for box in boxes)
    y_max = max(box['y'] + box['h'] for box in boxes)

    original_x_min = crop_area['left'] + x_min
    original_y_min = crop_area['top'] + y_min
    original_x_max = crop_area['left'] + x_max
    original_y_max = crop_area['top'] + y_max

    cropped_final_box = {
        "x": x_min,
        "y": y_min,
        "w": x_max - x_min,
        "h": y_max - y_min,
    }

    final_box = {
        "x": original_x_min,
        "y": original_y_min,
        "w": original_x_max - original_x_min,
        "h": original_y_max - original_y_min,
    }
    return cropped_final_box, final_box

def get_final_subtitle_mask(highlighted_image, original_image, cropped_final_box, crop_area):
    x, y, w, h = cropped_final_box['x'], cropped_final_box['y'], cropped_final_box['w'], cropped_final_box['h']
    mask = np.zeros_like(highlighted_image, dtype=np.uint8)
    roi_highlighted = highlighted_image[y:y+h, x:x+w]
    white_pixels_condition = np.all(roi_highlighted == [255, 255, 255], axis=2)
    mask[y:y+h, x:x+w][white_pixels_condition] = [255, 255, 255]

    # 원본 이미지에 맞게 마스크 조정
    original_mask = np.zeros_like(original_image, dtype=np.uint8)
    original_mask[crop_area['top'] + y:crop_area['top'] + y + h, crop_area['left'] + x:crop_area['left'] + x + w] = mask[y:y+h, x:x+w]

    return original_mask

def visualize_final_mask(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Final Subtitle Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
# all in one
def extract_subtitle_pixel_from_image(original_image, target_color, crop_area,
                                      pixel_highlighting_tolerances, psm_modes, conf_thresholds):
    cropped_image = crop_image(original_image, crop_area)

    highlight_results = highlight_subtitle_area_with_multi_tolerances(cropped_image, target_color, pixel_highlighting_tolerances)
    highlighted_image = highlight_results[9]['highlighted_image']
    all_boxes = find_boundingbox_from_highlighted_results(highlight_results, psm_modes, conf_thresholds)
    
    trimmed_boxes = boxes_trimming(highlighted_image, all_boxes, target_color, box_trimming_tolerance=30)

    if( len(trimmed_boxes) < 50):
        print("No boxes found after trimming. Exiting.")
        return None, None
    
    mode_boxes = find_ylength_mode_boxes(trimmed_boxes)

    cropped_bounding_box, final_bounding_box = get_final_bounding_box(mode_boxes, crop_area)
    
    final_subtitle_mask = get_final_subtitle_mask(highlighted_image, original_image, cropped_bounding_box, crop_area)

    return final_bounding_box, final_subtitle_mask

def extract_subtitle_pixel_from_video(video_path, target_color, crop_area,
                                      pixel_highlighting_tolerances, psm_modes, conf_thresholds):
    frames = load_video(video_path)
    bounding_boxes = []
    subtitle_masks = []
    for i, frame in enumerate(frames):
        if i % 100 == 0:
            print(f"Processing frame {i}/{len(frames)}")
        bounding_box, subtitle_mask = extract_subtitle_pixel_from_image(frame, target_color, crop_area,
                                                                       pixel_highlighting_tolerances, psm_modes, conf_thresholds)
        bounding_boxes.append(bounding_box)
        subtitle_masks.append(subtitle_mask)
        if bounding_box is not None and subtitle_mask is not None:
            print(f"Frame {i}: Bounding Box: {bounding_box}, Subtitle Mask shape: {subtitle_mask.shape}")
    print("Video processing complete.")
    return bounding_boxes, subtitle_masks
    
if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    video_path = './videos/Squid_game.mp4'# yaml
    
    frames = load_video(video_path) 
    sample_image = frames[1500]
    
    crop_range = {"top": 870, "bottom": 1000, "left": 120, "right": 1800} # yaml
    
    target_color = [255, 255, 255] # yaml

    pixel_highlighting_tolerances = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # yaml
    conf_thresholds = [0, 10, 50] # yaml
    psm_modes = {
        "psm6": '--oem 3 --psm 6 -l kor+eng',
        "psm8": '--oem 3 --psm 8 -l kor+eng',
        "psm10": '--oem 3 --psm 10 -l kor+eng'
    }
    
    # bounding_box, subtitle_mask = extract_subtitle_pixel_from_image(sample_image, target_color, crop_area=crop_range)
    
    # visualize_final_mask(sample_image, subtitle_mask)

    # bounding_boxes, subtitle_masks = extract_subtitle_pixel_from_video(video_path, target_color, crop_area=crop_range)
