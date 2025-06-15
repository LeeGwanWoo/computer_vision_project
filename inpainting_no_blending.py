import cv2
import numpy as np

def inpaint_subtitle_pipeline_no_blending(img_rgb, subtitle_mask, bounding_box,
                                          level=3, num_passes=3, radius_patch=10, 
                                          kernel_size=7, feather_sigma=1.0, threshold_schedule=[20,15,10,5], 
                                          fine_pass=True, fine_radius=5, fine_kernel=5, fine_feather=0.5, fine_threshold=5):
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Initial mask refine
    mask_u8 = subtitle_mask.astype(np.uint8) * 255
    mask_dil = cv2.dilate(mask_u8, kernel, iterations=2)
    mask_feather = cv2.GaussianBlur(mask_dil, (0, 0), feather_sigma)
    _, mask_refined = cv2.threshold(mask_feather, 50, 255, cv2.THRESH_BINARY)
    mask_refined_bool = (mask_refined == 255)

    # Start from original image
    recovered = img_rgb.copy()  # No blending
    current_img = recovered.copy()
    current_mask_bool = mask_refined_bool.copy()

    residual_mask_list = []
    
    y1, y2, x1, x2 = bounding_box

    # Multi-pass coarse inpaint
    for pass_idx in range(num_passes):
        residual_threshold_current = threshold_schedule[min(pass_idx, len(threshold_schedule)-1)]

        # Pyramid Down
        I_list = [current_img]
        M_list = [current_mask_bool.astype(np.uint8) * 255]

        for i in range(level):
            I_down = cv2.pyrDown(I_list[-1])
            M_down = cv2.resize(M_list[-1], (I_down.shape[1], I_down.shape[0]), interpolation=cv2.INTER_NEAREST)
            I_list.append(I_down)
            M_list.append(M_down)

        # Diffusion Inpaint @ lowest level
        I_lowest = I_list[-1]
        M_lowest = M_list[-1]

        I_lowest_bgr = cv2.cvtColor(I_lowest.astype(np.uint8), cv2.COLOR_RGB2BGR)  # 수정됨 (정상)
        inpaint_lowest_bgr = cv2.inpaint(I_lowest_bgr, M_lowest, radius_patch, cv2.INPAINT_TELEA)
        I_rec = cv2.cvtColor(inpaint_lowest_bgr, cv2.COLOR_BGR2RGB)

        # Pyramid Up
        for i in reversed(range(level)):
            I_up = cv2.pyrUp(I_rec)
            I_up = cv2.resize(I_up, (I_list[i].shape[1], I_list[i].shape[0]), interpolation=cv2.INTER_LINEAR)
            I_rec = I_up

        # subtitle_mask_dil 영역에만 적용 (안정적 적용)
        current_img[y1:y2, x1:x2] = I_rec[y1:y2, x1:x2]

        # Residual 계산 & 저장
        residual = current_img.astype(np.int16) - recovered.astype(np.int16)
        residual = np.abs(residual).astype(np.uint8)
        residual_gray = cv2.cvtColor(residual, cv2.COLOR_RGB2GRAY)
        _, residual_mask = cv2.threshold(residual_gray, residual_threshold_current, 255, cv2.THRESH_BINARY)

        residual_mask_dil = cv2.dilate(residual_mask, kernel, iterations=1)
        residual_mask_feather = cv2.GaussianBlur(residual_mask_dil, (0, 0), feather_sigma)
        _, residual_mask_refined = cv2.threshold(residual_mask_feather, 50, 255, cv2.THRESH_BINARY)

        current_mask_bool = (residual_mask_refined == 255)

        residual_mask_list.append(residual_mask_refined.copy())

    # Fine pass (level=1) optional
    if fine_pass:
        fine_mask_bool = (residual_mask_list[-1] == 255)
        fine_kernel_mat = np.ones((fine_kernel, fine_kernel), np.uint8)

        I_list = [current_img]
        M_list = [fine_mask_bool.astype(np.uint8) * 255]

        for i in range(1):  # level=1만 사용
            I_down = cv2.pyrDown(I_list[-1])
            M_down = cv2.resize(M_list[-1], (I_down.shape[1], I_down.shape[0]), interpolation=cv2.INTER_NEAREST)
            I_list.append(I_down)
            M_list.append(M_down)

        I_lowest = I_list[-1]
        M_lowest = M_list[-1]

        I_lowest_bgr = cv2.cvtColor(I_lowest.astype(np.uint8), cv2.COLOR_RGB2BGR)  # 수정됨 (정상)
        inpaint_lowest_bgr = cv2.inpaint(I_lowest_bgr, M_lowest, fine_radius, cv2.INPAINT_TELEA)
        I_rec = cv2.cvtColor(inpaint_lowest_bgr, cv2.COLOR_BGR2RGB)

        for i in reversed(range(1)):
            I_up = cv2.pyrUp(I_rec)
            I_up = cv2.resize(I_up, (I_list[i].shape[1], I_list[i].shape[0]), interpolation=cv2.INTER_LINEAR)
            I_rec = I_up

        # subtitle_mask_dil 영역에만 적용 (안정적 적용)
        current_img[y1:y2, x1:x2] = I_rec[y1:y2, x1:x2]

    return current_img, residual_mask_list

def apply_inpainting(frames, masks, boxes,
                     level, num_passes, radius_patch,
                     kernel_size, feather_sigma, threshold_schedule,
                     fine_pass, fine_radius, fine_kernel, fine_feather, fine_threshold):
    inpainted = []
    for f, m, bb in zip(frames, masks, boxes):
        if m is None or bb is None:
            inpainted.append(f)
        else:
            # bounding box 영역만 inpaint
            roi = f.copy()
            # mask는 full-frame mask이므로, inpaint 파이프라인에 넘기고
            recovered, _ = inpaint_subtitle_pipeline_no_blending(
                f, m[:, :, 0],  # full-frame image & mask
                (bb["y"], bb["y"]+bb["h"], bb["x"], bb["x"]+bb["w"]),  # (y1,y2,x1,x2)
                level, num_passes, radius_patch,
                kernel_size, feather_sigma, threshold_schedule,
                fine_pass, fine_radius, fine_kernel, fine_feather, fine_threshold
            )
            inpainted.append(recovered)
    return inpainted