#!/usr/bin/env python
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import os
import math
import argparse

# HSV Color Ranges
COLOR_RANGES = {
    'basketball': [
        {'lower': np.array([7, 140, 154]), 'upper': np.array([7, 152, 180])},
        {'lower': np.array([5, 100, 100]), 'upper': np.array([25, 255, 255])},
    ],
    'football': [
        {'lower': np.array([35, 20, 70]), 'upper': np.array([78, 112, 114])},
    ]
}

# Ball Filtering Parameters
BALL_FILTER_PARAMS = {
    'basketball': {
        'min_radius': 10, 'max_radius': 41, 'min_circularity': 0.65,
        'max_aspect_ratio_diff': 0.5, 'min_solidity': 0.75
    },
    'football': {
        'min_radius': 8, 'max_radius': 17, 'min_circularity': 0.55,
        'max_aspect_ratio_diff': 0.6, 'min_solidity': 0.65
    }
}

class KalmanBoxTracker:
    """Kalman Filter tracker for bounding boxes."""
    count = 0
    def __init__(self, bbox_xywh, frame, alpha_hist=0.1, default_process_noise=1e-2,
                 high_process_noise=5e-2, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],
                                           [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
                                          [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]], np.float32)
        self.default_process_noise_val = default_process_noise
        self.high_process_noise_val = high_process_noise
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * self.default_process_noise_val
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        x, y, w, h = bbox_xywh; w, h = max(1.0, w), max(1.0, h)
        cx, cy = x + w / 2.0, y + h / 2.0
        self.kf.statePre = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
        self.kf.errorCovPre = np.eye(8, dtype=np.float32) * 1.0
        self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1
        self.time_since_update = 0; self.hits = 1; self.hit_streak = 1; self.age = 0
        self.last_measurement_bbox = bbox_xywh
        self.color_hist = None; self.alpha_hist = alpha_hist
        if frame is not None and frame.size > 0:
             self.update_appearance(frame, self._get_bbox_xywh_from_state(self.kf.statePost))

    def _get_bbox_xywh_from_state(self, state):
        w, h = max(1.0, state[2, 0]), max(1.0, state[3, 0])
        cx, cy = state[0, 0], state[1, 0]
        return [cx - w / 2.0, cy - h / 2.0, w, h]

    def update_appearance(self, frame, bbox_xywh):
        if frame is None or frame.size == 0: return
        x, y, w, h = [int(v) for v in bbox_xywh]; img_h, img_w = frame.shape[:2]
        x, y = max(0, x), max(0, y); w, h = min(w, img_w - x), min(h, img_h - y)
        if w <= 0 or h <= 0: return
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: return
        try: hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error: return
        new_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(new_hist, new_hist, 0, 1, cv2.NORM_MINMAX); new_hist = new_hist.astype(np.float32)
        if self.color_hist is None: self.color_hist = new_hist
        else: self.color_hist = self.alpha_hist * new_hist + (1 - self.alpha_hist) * self.color_hist; cv2.normalize(self.color_hist, self.color_hist, 0, 1, cv2.NORM_MINMAX)

    def predict(self):
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1; self.age += 1
        prediction_state = self.kf.predict()
        predicted_bbox = self._get_bbox_xywh_from_state(prediction_state)
        noise_factor = 0.5 if self.hit_streak > 5 else 1.0
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * self.default_process_noise_val * noise_factor
        return predicted_bbox

    def update(self, bbox_xywh, frame):
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1
        self.last_measurement_bbox = bbox_xywh
        predicted_bbox = self._get_bbox_xywh_from_state(self.kf.statePre)
        pred_center = np.array([predicted_bbox[0] + predicted_bbox[2]/2.0, predicted_bbox[1] + predicted_bbox[3]/2.0])
        meas_center = np.array([bbox_xywh[0] + bbox_xywh[2]/2.0, bbox_xywh[1] + bbox_xywh[3]/2.0])
        measurement_error = np.linalg.norm(pred_center - meas_center)
        obj_size = max(1.0, max(bbox_xywh[2], bbox_xywh[3]))
        if (obj_size > 0) and (measurement_error / obj_size) > 0.5: self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * self.high_process_noise_val
        else: self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * self.default_process_noise_val
        x, y, w, h = bbox_xywh; w, h = max(1.0, w), max(1.0, h); cx, cy = x + w / 2.0, y + h / 2.0
        measurement = np.array([[cx], [cy], [w], [h]], np.float32)
        self.kf.correct(measurement)
        if frame is not None and frame.size > 0: self.update_appearance(frame, bbox_xywh)
        return self.get_state()

    def get_state(self): return self._get_bbox_xywh_from_state(self.kf.statePost)

def calculate_iou(bbox1_xywh, bbox2_xywh):
    if not (isinstance(bbox1_xywh, (list, tuple, np.ndarray)) and len(bbox1_xywh) == 4) or \
       not (isinstance(bbox2_xywh, (list, tuple, np.ndarray)) and len(bbox2_xywh) == 4): return 0.0
    x1, y1, w1, h1 = bbox1_xywh; x2, y2, w2, h2 = bbox2_xywh
    w1, h1 = max(0, w1), max(0, h1); w2, h2 = max(0, w2), max(0, h2)
    x_left, y_top = max(x1, x2), max(y1, y2); x_right, y_bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area, bbox2_area = w1 * h1, w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    return max(0.0, intersection_area / union_area) if union_area > 0 else 0.0

def calculate_histogram_similarity(hist1, hist2):
    if hist1 is None or hist2 is None or hist1.size == 0 or hist2.size == 0: return 0.0
    if hist1.shape != hist2.shape or hist1.dtype != hist2.dtype:
        try: hist2 = cv2.resize(hist2, hist1.shape[:2]).astype(hist1.dtype)
        except Exception: return 0.0
        if hist1.shape != hist2.shape: return 0.0
    hist1 = hist1.astype(np.float32); hist2 = hist2.astype(np.float32)
    return max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

class AdvancedTracker:
    """Manages multiple KalmanBoxTrackers."""
    def __init__(self, max_age=40, min_hits=1, lost_max_age=80):
        self.max_age_frames = int(max_age); self.min_hits = int(min_hits); self.lost_max_age_frames = int(lost_max_age)
        self.trackers = []; self.lost_trackers = []; self.frame_count = 0
    def set_frame_rate(self, fps): print(f"Tracker: max_age={self.max_age_frames}f, min_hits={self.min_hits}, lost_max={self.lost_max_age_frames}f")
    def update(self, detections_bboxes, frame):
        self.frame_count += 1
        detections = np.array(detections_bboxes) if detections_bboxes else np.empty((0, 4))
        if detections.ndim == 1 and detections.shape[0] == 4: detections = detections.reshape(1, 4)
        elif detections.ndim != 2 or (detections.shape[0] > 0 and detections.shape[1] != 4): detections = np.empty((0, 4))
        predicted_bboxes, temp_trackers = [], []
        for trk in self.trackers:
            pred_pos = trk.predict()
            if not np.any(np.isnan(pred_pos)) and not np.any(np.isinf(pred_pos)) and pred_pos[2] > 0 and pred_pos[3] > 0:
                predicted_bboxes.append(pred_pos); temp_trackers.append(trk)
        self.trackers = temp_trackers; predicted_bboxes = np.array(predicted_bboxes) if predicted_bboxes else np.empty((0, 4))
        matched_indices, unmatched_det_indices, unmatched_trk_indices = self.associate_detections_to_trackers(detections, predicted_bboxes, self.trackers, frame)
        updated_tracker_indices = set()
        for d_idx, t_idx in matched_indices:
            if t_idx < len(self.trackers):
                try: self.trackers[t_idx].update(detections[d_idx], frame); updated_tracker_indices.add(t_idx)
                except Exception: pass
        new_trackers_list, lost_trackers_list = [], self.lost_trackers
        for i, trk in enumerate(self.trackers):
            if i in updated_tracker_indices: new_trackers_list.append(trk)
            elif trk.time_since_update >= self.max_age_frames: lost_trackers_list.append(trk)
            else: new_trackers_list.append(trk)
        newly_created_trackers = []
        for i in unmatched_det_indices:
            if i < len(detections):
                det = detections[i]
                if det[2] > 0 and det[3] > 0:
                    try: newly_created_trackers.append(KalmanBoxTracker(det, frame))
                    except Exception: pass
        self.trackers = new_trackers_list + newly_created_trackers
        self.lost_trackers = [trk for trk in lost_trackers_list if trk.time_since_update < self.lost_max_age_frames]
        return_trackers = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                pos = trk.get_state()
                if not np.any(np.isnan(pos)) and not np.any(np.isinf(pos)) and pos[2] > 0 and pos[3] > 0:
                    return_trackers.append(np.append(pos, trk.id))
        return np.array(return_trackers) if return_trackers else np.empty((0, 5))
    def associate_detections_to_trackers(self, detections, predicted_bboxes, trackers_list, frame, iou_threshold=0.3, use_appearance=True, appearance_weight=0.3):
        num_detections, num_trackers = detections.shape[0], predicted_bboxes.shape[0]
        if num_trackers == 0: return [], np.arange(num_detections), []
        if num_detections == 0: return [], [], np.arange(num_trackers)
        iou_matrix = np.zeros((num_detections, num_trackers), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk_pred in enumerate(predicted_bboxes):
                if det[2] > 0 and det[3] > 0 and trk_pred[2] > 0 and trk_pred[3] > 0: iou_matrix[d, t] = calculate_iou(det, trk_pred)
        iou_cost_matrix = 1.0 - iou_matrix; cost_matrix = iou_cost_matrix
        if use_appearance and frame is not None and frame.size > 0 and len(trackers_list) == num_trackers:
            appearance_cost_matrix = np.ones_like(iou_cost_matrix)
            for d, det in enumerate(detections):
                d_x, d_y, d_w, d_h = [int(v) for v in det]; img_h, img_w = frame.shape[:2]; d_x, d_y = max(0, d_x), max(0, d_y); d_w, d_h = min(d_w, img_w - d_x), min(d_h, img_h - d_y)
                if d_w <= 0 or d_h <= 0: continue
                d_roi = frame[d_y:d_y+d_h, d_x:d_x+d_w]; d_hist = None
                if d_roi.size > 0:
                    try: d_hsv = cv2.cvtColor(d_roi, cv2.COLOR_BGR2HSV); d_hist = cv2.calcHist([d_hsv], [0], None, [180], [0, 180]); cv2.normalize(d_hist, d_hist, 0, 1, cv2.NORM_MINMAX); d_hist = d_hist.astype(np.float32)
                    except cv2.error: d_hist = None
                if d_hist is None: continue
                for t, trk in enumerate(trackers_list):
                    if trk.color_hist is not None and trk.color_hist.size > 0: appearance_cost_matrix[d, t] = 1.0 - calculate_histogram_similarity(d_hist, trk.color_hist)
            cost_matrix = (1.0 - appearance_weight) * iou_cost_matrix + appearance_weight * appearance_cost_matrix
        try: row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError: return [], np.arange(num_detections), np.arange(num_trackers)
        matches, unmatched_detections, unmatched_trackers = [], list(range(num_detections)), list(range(num_trackers))
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c));
                if r in unmatched_detections: unmatched_detections.remove(r)
                if c in unmatched_trackers: unmatched_trackers.remove(c)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def get_background_subtractor_mask(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame); kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)); kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask_cleaned = cv2.erode(fg_mask, kernel_erode, iterations=1); fg_mask_cleaned = cv2.dilate(fg_mask_cleaned, kernel_dilate, iterations=2)
    _, fg_mask_binary = cv2.threshold(fg_mask_cleaned, 128, 255, cv2.THRESH_BINARY); return fg_mask_binary

def get_frame_diff_mask(frame, prev_frame, threshold=25):
    if prev_frame is None: return np.zeros(frame.shape[:2], dtype=np.uint8)
    gray_diff = cv2.cvtColor(cv2.absdiff(prev_frame, frame), cv2.COLOR_BGR2GRAY); _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2); return mask

def detect_balls_by_color(frame, video_type):
    color_ranges = COLOR_RANGES.get(video_type, []); combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if not color_ranges: return [], combined_mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for color_range in color_ranges: combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, color_range['lower'], color_range['upper']))
    if cv2.countNonZero(combined_mask) == 0: return [], combined_mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)); kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1), cv2.MORPH_OPEN, kernel_open, iterations=1)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); return filter_ball_candidates(contours, frame, video_type), cleaned_mask

def filter_ball_candidates(contours, frame, video_type='basketball'):
    ball_detections_conf = []; frame_h, _ = frame.shape[:2]; params = BALL_FILTER_PARAMS.get(video_type, BALL_FILTER_PARAMS['basketball'])
    min_area, max_area = math.pi * (params['min_radius'] ** 2), math.pi * (params['max_radius'] ** 2); min_conf_thresh = 0.1
    for c in contours:
        area = cv2.contourArea(c);
        if not (min_area < area < max_area): continue
        x, y, w, h = cv2.boundingRect(c);
        if h == 0 or w == 0: continue
        aspect_ratio = float(w) / h;
        if abs(aspect_ratio - 1.0) > params['max_aspect_ratio_diff']: continue
        hull = cv2.convexHull(c); hull_area = cv2.contourArea(hull);
        if hull_area == 0: continue
        solidity = float(area) / hull_area;
        if solidity < params['min_solidity']: continue
        perimeter = cv2.arcLength(c, True);
        if perimeter == 0: continue
        circularity = 4 * math.pi * area / (perimeter * perimeter);
        if circularity < params['min_circularity']: continue
        conf_circ = max(0.0, 1.0 - abs(circularity - 1.0)); solidity_range = 1.0 - params['min_solidity']
        conf_solid = min(1.0, max(0.0, (solidity - params['min_solidity']) / solidity_range if solidity_range > 0 else solidity)); aspect_range = params['max_aspect_ratio_diff']
        conf_aspect = min(1.0, max(0.0, 1.0 - (abs(aspect_ratio - 1.0) / aspect_range if aspect_range > 0 else (0.0 if aspect_ratio != 1.0 else 1.0))))
        confidence_score = (conf_circ + conf_solid + conf_aspect) / 3.0
        if video_type == 'football' and (y + h / 2.0) < frame_h * 0.30: confidence_score *= 0.8
        if confidence_score >= min_conf_thresh: ball_detections_conf.append(([x, y, w, h], confidence_score))
    return ball_detections_conf

def merge_detections(color_dets_conf, motion_dets_conf, iou_thresh=0.15, color_weight=0.75):
    if not color_dets_conf and not motion_dets_conf: return []
    if not color_dets_conf: return motion_dets_conf
    if not motion_dets_conf: return color_dets_conf
    c_bboxes = [d[0] for d in color_dets_conf]; c_confs = [d[1] for d in color_dets_conf]; m_bboxes = [d[0] for d in motion_dets_conf]; m_confs = [d[1] for d in motion_dets_conf]
    used_motion_indices = set(); final_dets_conf = []
    for i, c_bbox in enumerate(c_bboxes):
        c_conf = c_confs[i]; best_iou, best_m_idx = -1.0, -1
        for j, m_bbox in enumerate(m_bboxes):
            if j not in used_motion_indices:
                iou = calculate_iou(c_bbox, m_bbox);
                if iou >= iou_thresh and iou > best_iou: best_iou, best_m_idx = iou, j
        if best_m_idx != -1:
            used_motion_indices.add(best_m_idx); m_bbox = m_bboxes[best_m_idx]
            merged_x = c_bbox[0] * color_weight + m_bbox[0] * (1 - color_weight); merged_y = c_bbox[1] * color_weight + m_bbox[1] * (1 - color_weight)
            merged_w = c_bbox[2] * color_weight + m_bbox[2] * (1 - color_weight); merged_h = c_bbox[3] * color_weight + m_bbox[3] * (1 - color_weight)
            merged_conf = min(1.0, c_conf * (1 + best_iou * 0.2)); final_dets_conf.append(([merged_x, merged_y, merged_w, merged_h], merged_conf))
        else: final_dets_conf.append((c_bbox, c_conf))
    for j in range(len(m_bboxes)):
        if j not in used_motion_indices: final_dets_conf.append((m_bboxes[j], m_confs[j]))
    return final_dets_conf

def suppress_reflections(mask, frame, video_type):
    if video_type != 'basketball' or frame is None or frame.size == 0: return mask
    h, _ = frame.shape[:2]; reflection_zone_mask = np.zeros_like(mask); reflection_zone_mask[int(h * 0.60):, :] = 255
    lower_floor, upper_floor = np.array([10, 50, 50]), np.array([35, 180, 200])
    try: floor_color_mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_floor, upper_floor)
    except cv2.error: return mask
    potential_ref_mask = cv2.bitwise_and(reflection_zone_mask, floor_color_mask); potential_ref_mask = cv2.dilate(potential_ref_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
    return cv2.bitwise_and(mask, cv2.bitwise_not(potential_ref_mask))

def visualize_tracking(frame, tracked_balls, trajectories, frame_count, mask=None):
    vis_frame = frame.copy()
    if mask is not None and mask.ndim == 2 and mask.size > 0:
        try:
             mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
             if mask_colored.shape == vis_frame.shape:
                 vis_frame = cv2.addWeighted(vis_frame, 0.7, mask_colored, 0.3, 0)
        except cv2.error as e:
             print(f"Error during mask overlay visualization: {e}")
    if tracked_balls is not None and tracked_balls.ndim == 2 and tracked_balls.shape[1] == 5:
        for x, y, w, h, ball_id in tracked_balls:
            x, y, w, h, ball_id = int(x), int(y), int(w), int(h), int(ball_id)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"B{ball_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            center = (int(x + w / 2.0), int(y + h / 2.0))
            if ball_id not in trajectories: trajectories[ball_id] = []
            trajectories[ball_id].append(center); max_traj_len = 40
            trajectories[ball_id] = trajectories[ball_id][-max_traj_len:]
            for i in range(1, len(trajectories[ball_id])):
                if trajectories[ball_id][i - 1] is not None and trajectories[ball_id][i] is not None:
                    thickness = max(1, int(np.sqrt(max_traj_len / float(i + 1)) * 1.5))
                    cv2.line(vis_frame, trajectories[ball_id][i - 1], trajectories[ball_id][i], (0, 255, 0), thickness)
    num_tracked = len(tracked_balls) if tracked_balls is not None and tracked_balls.shape[0] > 0 else 0
    info_text = f"Frame: {frame_count} | Tracked: {num_tracked}"; cv2.putText(vis_frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis_frame

def process_video(video_path, video_type, output_path=None, display=True, top_n_balls=2, min_hits=2, max_age=30, lost_max_age=60):
    """Main function to process a video file for ball detection and tracking."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: Could not open video file: {video_path}"); return
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS); fps = 30.0 if fps is None or fps <= 0 else fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing: {os.path.basename(video_path)} ({width}x{height} @ {fps:.2f} FPS, {total_frames if total_frames > 0 else '?'} Frames)")
    out = None
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True); fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height));
            if not out.isOpened(): print(f"Error: Could not open VideoWriter for path: {output_path}. Output disabled."); out = None
        except Exception as e:
            print(f"Error initializing VideoWriter: {e}. Output disabled."); out = None

    KalmanBoxTracker.count = 0 # Reset tracker ID counter for each video
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
    ball_tracker = AdvancedTracker(max_age=max_age, min_hits=min_hits, lost_max_age=lost_max_age); ball_tracker.set_frame_rate(fps)
    ball_trajectories = {}; frame_count, prev_frame, processing_times = 0, None, []
    display_active = display # Store initial display preference

    while True:
        ret, frame = cap.read()
        if not ret: print("\nEnd of video or cannot read frame."); break
        if frame is None or frame.size == 0: print(f"Warning: Received empty frame at index {frame_count}. Skipping."); continue
        frame_count += 1; start_time = time.time()
        motion_mask_raw = cv2.bitwise_or(get_background_subtractor_mask(frame, bg_subtractor), get_frame_diff_mask(frame, prev_frame, threshold=25))
        motion_mask_final = suppress_reflections(motion_mask_raw, frame, video_type)
        motion_contours, _ = cv2.findContours(motion_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_dets_conf = filter_ball_candidates(motion_contours, frame, video_type)
        color_dets_conf, _ = detect_balls_by_color(frame, video_type)
        merged_dets_conf = merge_detections(color_dets_conf, motion_dets_conf, iou_thresh=0.15, color_weight=0.75)
        merged_dets_conf.sort(key=lambda x: x[1], reverse=True)
        top_bboxes = [det[0] for det in merged_dets_conf[:top_n_balls]]
        tracked_balls = ball_tracker.update(top_bboxes, frame)
        processing_times.append(time.time() - start_time)
        vis_frame = visualize_tracking(frame, tracked_balls, ball_trajectories, frame_count, mask=None)
        if display_active:
            try:
                cv2.imshow(f"Tracking - {video_type}", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): print("Exit requested by user ('q' pressed)."); break
            except cv2.error as e:
                 if "DISPLAY" in str(e) or "cannot connect" in str(e).lower() or "GTK" in str(e): print("Display unavailable (No GUI environment?). Disabling display."); display_active = False
                 else: print(f"Unexpected cv2.imshow error: {e}"); break
        if out is not None:
            try: out.write(vis_frame)
            except Exception as e: print(f"Error writing frame {frame_count} to output video: {e}"); out.release(); out = None
        prev_frame = frame.copy()
        if frame_count % 100 == 0 and processing_times:
            last_100_times = processing_times[-100:]; avg_fps_100 = len(last_100_times) / sum(last_100_times) if sum(last_100_times) > 0 else 0
            prog_str = f"Frame {frame_count}" + (f"/{total_frames}" if total_frames > 0 else "") + f" | Avg FPS (last 100): {avg_fps_100:.2f}"; print(prog_str, end='\r')

    print("\nReleasing video capture..."); cap.release()
    if out is not None: print(f"Releasing video writer: {output_path}"); out.release()
    if display_active:
        print("Closing OpenCV display windows...");
        try: cv2.destroyAllWindows(); cv2.waitKey(1)
        except Exception as e: print(f"Error during cv2.destroyAllWindows: {e}")
    if processing_times:
        avg_time_per_frame = sum(processing_times) / len(processing_times); overall_avg_fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0
        print(f"\n--- Processing Summary: {os.path.basename(video_path)} ---"); print(f"Total Frames Processed: {frame_count}"); print(f"Average Time per Frame: {avg_time_per_frame:.4f} seconds"); print(f"Overall Average FPS: {overall_avg_fps:.2f}")
        if output_path and os.path.exists(output_path): print(f"Output video saved to: {output_path}")
        elif output_path: print(f"Output video specified but NOT found (likely failed): {output_path}")
        print("-" * 40)
    else: print("No frames were processed.")

def main():
    """Sets up configuration and runs the video processing via arguments."""
    parser = argparse.ArgumentParser(description="Detect and track balls in a video.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--video_type", required=True, choices=['basketball', 'football'], help="Type of sport in the video.")
    parser.add_argument("--output_path", "-o", default=None, help="Path to save the output video with tracking visualization. (e.g., output/tracked.mp4)")
    parser.add_argument("--no_display", action="store_true", help="Disable interactive display window.")
    parser.add_argument("--top_n_balls", type=int, default=None, help="Number of top confident detections to track per frame. Defaults: 1 for basketball, 2 for football.")
    parser.add_argument("--min_hits", type=int, default=None, help="Minimum consecutive hits to confirm a track. Defaults: 1 for basketball, 2 for football.")
    parser.add_argument("--max_age", type=int, default=30, help="Maximum number of frames a track can survive without updates. Default: 30")
    parser.add_argument("--lost_max_age", type=int, default=60, help="Maximum number of frames to keep a track after it's marked as lost. Default: 60")

    args = parser.parse_args()

    # Set default top_n and min_hits based on video_type if not provided
    if args.top_n_balls is None:
        args.top_n_balls = 1 if args.video_type == 'basketball' else 2
    if args.min_hits is None:
        args.min_hits = 1 if args.video_type == 'basketball' else 2

    if not os.path.exists(args.video_path):
        print(f"ERROR: Input video not found: {args.video_path}"); return

    if args.output_path:
        try:
            out_dir = os.path.dirname(args.output_path)
            if out_dir and not os.path.exists(out_dir):
                 os.makedirs(out_dir, exist_ok=True)
                 print(f"Created output directory: {out_dir}")
        except OSError as e:
            print(f"ERROR: Could not create output directory for {args.output_path}: {e}")
            # Decide if you want to stop or just disable output
            # return
            print("Warning: Output path directory creation failed. Disabling video output.")
            args.output_path = None

    print(f"\n>>> Processing {args.video_path} (Type: {args.video_type})")
    print(f"Settings: Display={not args.no_display}, Top N={args.top_n_balls}, Min Hits={args.min_hits}, Max Age={args.max_age}, Lost Max Age={args.lost_max_age}")
    if args.output_path:
        print(f"Output video: {args.output_path}")

    process_video(
        video_path=args.video_path,
        video_type=args.video_type,
        output_path=args.output_path,
        display=not args.no_display,
        top_n_balls=args.top_n_balls,
        min_hits=args.min_hits,
        max_age=args.max_age,
        lost_max_age=args.lost_max_age
    )
    print("\nProcessing complete.")

if __name__ == "__main__":
    
    main()
