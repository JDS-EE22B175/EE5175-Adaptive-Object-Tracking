# -----------------------------------------------------------------------------
# Object Detection and Tracking (Python File Version - Hybrid Ball Detection)
#
# Based on: Kalman Filter, Hungarian Algorithm, Appearance Matching.
# Detection:
#   - 'all' mode: Background Subtraction + Optical Flow motion mask.
#   - 'balls' mode: **Hybrid: Color Segmentation + Motion Detection, merged.**
# Features: Handles occlusions, optional ball filtering, noise/aspect ratio checks.
#           Improved ball detection reliability, including fast motion.
# Usage:    Run from command line (see __main__ block).
#           Press 'q' or 'ESC' to quit, 'p' to pause/resume.
# Requires: opencv-python, numpy, scipy
# -----------------------------------------------------------------------------

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import os
import argparse # For command-line arguments

# --- Constants and Tunable Parameters ---

# Tracking parameters
MAX_AGE_SECONDS = 30         # Max duration (seconds) a track persists without updates
MIN_HITS_TO_CONFIRM = 1      # Min consecutive detections to confirm
APPEARANCE_WEIGHT = 0.5     # Weight for appearance cost in association
COST_MATCH_THRESHOLD = 0.8   # Max combined cost for a valid match
IOU_GATE_THRESHOLD = 0.8     # Max IoU cost (1-IoU) allowed before assignment (min IoU ~0.2)

# Detection parameters (Adaptive Background Subtractor - used in 'all' AND 'balls' mode)
BGS_HISTORY = 200
BGS_VAR_THRESHOLD = 20
BGS_LEARNING_RATE = 0.005    # Slower adaptation

# Detection parameters (Optical Flow - used in 'all' AND 'balls' mode now)
OPTFLOW_SCALE = 0.4
OPTFLOW_MAG_THRESHOLD = 1.8

# Detection parameters (Color Segmentation - used in 'balls' mode)
# ====> !!! CRITICAL: TUNE THESE HSV RANGES FOR YOUR VIDEOS/BALLS/LIGHTING !!! <====
COLOR_RANGES = {
    'ball_orange_brown': ([5, 100, 100], [25, 255, 255]),
    'ball_white': ([0, 0, 170], [180, 70, 255]),
    'ball_black': ([0, 0, 90], [180, 60, 180])
}
COLOR_MASK_CLOSE_KERNEL_SIZE = (9, 9)
COLOR_MASK_OPEN_KERNEL_SIZE = (5, 5)

# Detection parameters (Contour Filtering - 'all' mode - applied AFTER motion mask)
FILTER_MIN_ABS_AREA = 100
FILTER_MIN_REL_AREA_PCT = 0.0005
FILTER_MIN_ASPECT_RATIO = 0.15
FILTER_MAX_ASPECT_RATIO = 6.0
FILTER_MIN_SOLIDITY = 0.60

# Detection parameters (Contour Filtering - 'balls' mode - applied AFTER color AND motion masks)
BALL_MIN_HEIGHT_PCT = 0.015
BALL_MAX_HEIGHT_PCT = 0.15
BALL_MIN_ABS_AREA = 10
BALL_ASPECT_RATIO_TOL = 0.5
BALL_MIN_CIRCULARITY = 0.5

# Merging detections from color and motion in 'balls' mode
DETECTION_MERGE_IOU_THRESHOLD = 0.1 # Min IoU to consider detections as the same ball

# Visualization
TRAJECTORY_LENGTH = 40

# --- KalmanBoxTracker Class ---

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, frame):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],
                                              [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.06
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        x, y, w, h = bbox
        self.kf.statePost = np.array([[x],[y],[w],[h],[0],[0],[0],[0]], np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1
        self.time_since_update = 0; self.hits = 1; self.hit_streak = 1; self.age = 0
        self.color_hist = np.zeros((180, 1), dtype=np.float32)
        self.update_appearance(frame, bbox)
        self.last_prediction = self.get_state()
    def update_appearance(self, frame, bbox):
        x,y,w,h=[int(v) for v in bbox]; x,y=max(0,x),max(0,y); w,h=min(w,frame.shape[1]-x-1),min(h,frame.shape[0]-y-1)
        if w > 0 and h > 0:
            roi=frame[y:y+h,x:x+w]; hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            self.color_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180]); cv2.normalize(self.color_hist,self.color_hist,0,1,cv2.NORM_MINMAX)
    def predict(self):
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1; self.age += 1
        prediction = self.kf.predict(); self.last_prediction = prediction[:4].flatten()
        return self.last_prediction
    def update(self, bbox, frame):
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1
        self.update_appearance(frame, bbox)
        measurement = np.array([[bbox[0]],[bbox[1]],[bbox[2]],[bbox[3]]], np.float32)
        self.kf.correct(measurement)
        return self.get_state()
    def get_state(self):
        return self.kf.statePost[:4].flatten()

# --- Utility Functions ---
def calculate_iou(bbox1, bbox2):
    x1_i=max(bbox1[0],bbox2[0]); y1_i=max(bbox1[1],bbox2[1]); x2_i=min(bbox1[0]+bbox1[2],bbox2[0]+bbox2[2]); y2_i=min(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])
    inter=max(0,x2_i-x1_i)*max(0,y2_i-y1_i); area1=bbox1[2]*bbox1[3]; area2=bbox2[2]*bbox2[3]; union=area1+area2-inter
    return inter/union if union>0 else 0.0

def calculate_histogram_similarity(hist1, hist2):
    if hist1 is None or hist2 is None or hist1.size == 0 or hist2.size == 0: return 0.0
    return max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

# --- AdvancedTracker Class ---

class AdvancedTracker:
    def __init__(self, max_age_seconds=MAX_AGE_SECONDS, min_hits=MIN_HITS_TO_CONFIRM,
                 appearance_weight=APPEARANCE_WEIGHT, cost_match_threshold=COST_MATCH_THRESHOLD,
                 iou_gate_threshold=IOU_GATE_THRESHOLD):
        self.max_age_seconds=max_age_seconds; self.max_age_frames=None; self.min_hits=min_hits
        self.appearance_weight=np.clip(appearance_weight,0.0,1.0); self.cost_match_threshold=cost_match_threshold
        self.iou_gate_threshold=iou_gate_threshold; self.trackers=[]; self.frame_count=0; KalmanBoxTracker.count=0
    def _set_max_age_frames(self, fps):
        if fps>0: self.max_age_frames=int(self.max_age_seconds*fps)
        else: self.max_age_frames=int(self.max_age_seconds*30)
        if self.max_age_frames<=0: self.max_age_frames=30
        print(f"[Tracker Info] Max track age set to {self.max_age_frames} frames ({self.max_age_seconds}s @ {fps:.1f} FPS)")
    def update(self, detections, frame, fps):
        self.frame_count+=1
        if self.max_age_frames is None: self._set_max_age_frames(fps)
        predicted_bboxes=np.array([trk.predict() for trk in self.trackers]) if self.trackers else np.empty((0,4))
        matched,unmatched_dets,unmatched_trks=self.associate_detections_to_trackers(detections,predicted_bboxes,frame)
        for di,ti in matched:
            if ti<len(self.trackers): self.trackers[ti].update(detections[di],frame)
        for di in unmatched_dets:
            x,y,w,h=detections[di]
            if w>0 and h>0: self.trackers.append(KalmanBoxTracker(detections[di],frame))
        active=[]; indices_to_pop=[]
        for i,trk in enumerate(self.trackers):
            state=trk.get_state() if trk.time_since_update==0 else trk.last_prediction
            is_confirmed=(trk.hit_streak>=self.min_hits or self.frame_count<=self.min_hits)
            if is_confirmed and trk.time_since_update<self.max_age_frames: active.append(np.append(state,trk.id))
            if trk.time_since_update>=self.max_age_frames: indices_to_pop.append(i)
        for i in sorted(indices_to_pop,reverse=True): self.trackers.pop(i)
        return np.array(active) if active else np.empty((0,5))
    def associate_detections_to_trackers(self, detections, predicted_bboxes, frame):
        n_dets,n_trks=detections.shape[0],predicted_bboxes.shape[0]
        if n_trks==0: return np.empty((0,2),dtype=int),np.arange(n_dets),[]
        if n_dets==0: return np.empty((0,2),dtype=int),[],np.arange(n_trks)
        iou_cost=np.zeros((n_dets,n_trks),dtype=np.float32); app_cost=np.ones((n_dets,n_trks),dtype=np.float32)
        det_hists=[self._get_det_hist(det,frame) for det in detections]
        for d in range(n_dets):
            for t in range(n_trks):
                iou_cost[d,t]=1.0-calculate_iou(detections[d],predicted_bboxes[t])
                if t<len(self.trackers): app_cost[d,t]=1.0-calculate_histogram_similarity(det_hists[d],self.trackers[t].color_hist)
        cost_matrix=((1.0-self.appearance_weight)*iou_cost)+(self.appearance_weight*app_cost)
        cost_matrix[iou_cost>self.iou_gate_threshold]=100.0
        r_ind,c_ind=linear_sum_assignment(cost_matrix); opt_matches=np.column_stack((r_ind,c_ind))
        matches=[]; un_dets=set(range(n_dets)); un_trks=set(range(n_trks))
        for r,c in opt_matches:
            if cost_matrix[r,c]<self.cost_match_threshold:
                matches.append([r,c]); un_dets.discard(r); un_trks.discard(c)
        return np.array(matches),np.array(list(un_dets)),np.array(list(un_trks))
    def _get_det_hist(self, det, frame):
        x,y,w,h=[int(v) for v in det]; x,y=max(0,x),max(0,y); w,h=min(w,frame.shape[1]-x-1),min(h,frame.shape[0]-y-1)
        hist=np.zeros((180,1),dtype=np.float32)
        if w>0 and h>0:
            roi=frame[y:y+h,x:x+w]; hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([hsv],[0],None,[180],[0,180]); cv2.normalize(hist,hist,0,1,cv2.NORM_MINMAX)
        return hist

# --- Background Subtractor & Motion Segmentation ---

class AdaptiveBackgroundSubtractor:
    def __init__(self, history=BGS_HISTORY, var_threshold=BGS_VAR_THRESHOLD, learning_rate=BGS_LEARNING_RATE, update_interval=5):
        self.bg=cv2.createBackgroundSubtractorMOG2(history=history,varThreshold=var_threshold,detectShadows=False)
        self.lr=learning_rate; self.vt=var_threshold; self.hist=history; self.last_upd=time.time(); self.upd_int=update_interval
        print(f"[Detector Info] BGS: History={history}, VarThresh={var_threshold}, LR={learning_rate:.4f}")
    def apply(self, frame):
        mask=self.bg.apply(frame,learningRate=self.lr); t=time.time()
        if t-self.last_upd>self.upd_int: self.adapt(frame,mask); self.last_upd=t
        return mask
    def adapt(self, frame, mask):
        pct=np.sum(mask>0)/mask.size; new_lr=max(0.0005,self.lr*0.9) if pct>0.3 else min(0.01,self.lr*1.1)
        if abs(new_lr-self.lr)>1e-5: self.lr=new_lr
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); _,std=cv2.meanStdDev(gray); new_vt=np.clip(std[0][0]*1.5,15,45)
        if abs(new_vt-self.vt)>1.0: self.vt=new_vt
        self.bg=cv2.createBackgroundSubtractorMOG2(history=self.hist,varThreshold=self.vt,detectShadows=False)

def motion_based_segmentation(frame, prev_frame, scale=OPTFLOW_SCALE):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); prev=cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape; gs=cv2.resize(gray,(int(w*scale),int(h*scale))); pgs=cv2.resize(prev,(int(w*scale),int(h*scale)))
    flow=cv2.calcOpticalFlowFarneback(pgs,gs,None,0.5,3,15,3,5,1.2,0)
    mag,_=cv2.cartToPolar(flow[...,0],flow[...,1]); mask_s=(mag>OPTFLOW_MAG_THRESHOLD).astype(np.uint8)*255
    mask=cv2.resize(mask_s,(w,h),interpolation=cv2.INTER_NEAREST); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=1)

# --- Contour Filtering Functions ---

def filter_contours_adaptive(contours, frame_size_hw):
    filt=[]; fh,fw=frame_size_hw; area_f=fh*fw; min_a=max(FILTER_MIN_ABS_AREA,area_f*FILTER_MIN_REL_AREA_PCT)
    for c in contours:
        area=cv2.contourArea(c)
        if area<min_a: continue
        x,y,w,h=cv2.boundingRect(c)
        if w<=0 or h<=0: continue
        aspect=float(w)/h; hull=cv2.convexHull(c); hull_area=cv2.contourArea(hull); solidity=area/hull_area if hull_area>0 else 0
        if not (FILTER_MIN_ASPECT_RATIO<aspect<FILTER_MAX_ASPECT_RATIO): continue
        if solidity<FILTER_MIN_SOLIDITY: continue
        filt.append(c)
    return filt

def filter_contours_for_balls(contours, frame_size_hw):
    filt=[]; fh,fw=frame_size_hw; min_h=fh*BALL_MIN_HEIGHT_PCT; max_h=fh*BALL_MAX_HEIGHT_PCT
    for c in contours:
        area=cv2.contourArea(c);
        if area<BALL_MIN_ABS_AREA: continue
        x,y,w,h=cv2.boundingRect(c);
        if w<=0 or h<=0: continue
        if not (min_h<h<max_h and min_h<w<max_h): continue
        aspect=float(w)/h;
        if abs(aspect-1.0)>BALL_ASPECT_RATIO_TOL: continue
        (_,radius)=cv2.minEnclosingCircle(c);
        if radius<=0: continue
        circ_area=np.pi*(radius**2); circularity=area/circ_area if circ_area>0 else 0
        if circularity<BALL_MIN_CIRCULARITY: continue
        filt.append(c)
    return filt

# --- Detection Merging Function ('balls' mode) ---
def merge_detections(detections1, detections2, iou_threshold=DETECTION_MERGE_IOU_THRESHOLD):
    """
    Merges two lists of bounding box detections based on IoU overlap.
    Prefers boxes from detections1 if overlap occurs.

    Args:
        detections1 (list): List of primary detections [(x,y,w,h), ...].
        detections2 (list): List of secondary detections [(x,y,w,h), ...].
        iou_threshold (float): Minimum IoU to consider boxes as overlapping.

    Returns:
        list: A merged list of unique detections [(x,y,w,h), ...].
    """
    if not detections1: return detections2
    if not detections2: return detections1

    final_detections = list(detections1) # Start with all primary detections
    used_det2 = [False] * len(detections2)

    for i, box1 in enumerate(detections1):
        for j, box2 in enumerate(detections2):
            if used_det2[j]: continue
            iou = calculate_iou(box1, box2)
            if iou > iou_threshold:
                used_det2[j] = True # Mark secondary detection as matched/merged

    # Add secondary detections that were NOT matched to any primary detection
    for j, box2 in enumerate(detections2):
        if not used_det2[j]:
            final_detections.append(box2)

    # Optional: Apply non-max suppression or further merging if needed,
    # but simple filtering might be enough here.
    return final_detections

# --- Visualization ---

def visualize_tracking(frame, tracked_objects, trajectories, frame_count, label="Objects"):
    vis=frame.copy()
    for x,y,w,h,id_ in tracked_objects:
        x,y,w,h,id_=int(x),int(y),int(w),int(h),int(id_)
        np.random.seed(id_); color=tuple(np.random.randint(60,255,size=3).tolist())
        cv2.rectangle(vis,(x,y),(x+w,y+h),color,2); lbl=f"ID:{id_}"; (tw,th),bl=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        cv2.rectangle(vis,(x,y-th-bl-2),(x+tw,y),color,-1); cv2.putText(vis,lbl,(x,y-bl),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        center=(int(x+w/2),int(y+h/2));
        if id_ not in trajectories: trajectories[id_]=[]
        trajectories[id_].append(center);
        if len(trajectories[id_])>TRAJECTORY_LENGTH: trajectories[id_]=trajectories[id_][-TRAJECTORY_LENGTH:]
        pts=trajectories[id_]
        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None: continue
            cv2.line(vis,pts[i-1],pts[i],color,2)
    info=f"Frame:{frame_count}|{label}:{len(tracked_objects)}"; cv2.putText(vis,info,(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    return vis

# --- Video Saving Helper Function ---
def initialize_video_writer(output_path, fps, frame_size_wh):
    """ Initializes and returns a cv2.VideoWriter object or None on error. """
    if not output_path:
        return None

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"[Error] Failed to create output directory '{output_dir}': {e}")
            return None

    # Ensure output filename has a common extension like .mp4
    if not output_path.lower().endswith(('.mp4', '.avi', '.mov')):
         output_path += '.mp4'
         print(f"Appending .mp4 to output filename: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v generally works well
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size_wh) # Use (width, height) for writer

    if not writer.isOpened():
        print(f"[Error] Could not open VideoWriter for path: {output_path}")
        return None
    else:
        print(f"Output video will be saved to: {output_path}")
        return writer

# --- Main Video Processing Function ---
def process_video(video_path, filter_mode='all', output_path=None, display=True, max_frames=None, show_mask=False):
    """ Main function to process video, perform detection/tracking, display/save. """
    # --- Setup & Initialization ---
    print(f"--- Processing Video ---"); print(f"Source: {video_path}"); print(f"Filter Mode: {filter_mode}");
    if output_path: print(f"Output Path: {output_path}")
    if not display: print("Display Disabled")
    if max_frames: print(f"Max Frames: {max_frames}")
    print("-" * 25)
    try: source=int(video_path)
    except ValueError: source=video_path;
    if isinstance(source, str) and not os.path.exists(source): print(f"[Error] Input video file not found: {source}"); return
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): print(f"[Error] Could not open video source: {source}"); return
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS); f_size_hw=(h,w); f_size_wh=(w,h)
    tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    if fps<=0: fps=30.0
    print(f"Input: {w}x{h} @ ~{fps:.1f} FPS ({tf if tf>0 else 'Unknown'} frames)")

    # Initialize video writer using helper function
    out_writer = initialize_video_writer(output_path, fps, f_size_wh)

    tracker=AdvancedTracker()
    # Initialize BGS for BOTH modes now, as it's used in the hybrid approach
    bg_sub = AdaptiveBackgroundSubtractor()
    trajectories={}; f_count=0
    ret, prev_frame=cap.read()
    if not ret: print("[Error] Failed to read first frame."); cap.release(); return
    start_time=time.time(); paused=False; vis=prev_frame.copy()

    # --- Main Processing Loop ---
    while cap.isOpened():
        if not paused:
            ret, frame=cap.read()
            if not ret: break
            f_count+=1; loop_start=time.time()
            if max_frames is not None and f_count > max_frames: print(f"\nReached max_frames ({max_frames})."); break

            # --- Detection (Mode Dependent) ---
            filt_contours = []; thresh_mask_for_display = None; lbl = "N/A"

            # ** Always run motion detection **
            fg=bg_sub.apply(frame); mo=motion_based_segmentation(frame,prev_frame); motion_combined=cv2.bitwise_or(fg,mo)
            ko=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); mc=cv2.morphologyEx(motion_combined,cv2.MORPH_OPEN,ko,iterations=1)
            kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)); mc=cv2.morphologyEx(mc,cv2.MORPH_CLOSE,kc,iterations=2)
            _,motion_thresh=cv2.threshold(mc,150,255,cv2.THRESH_BINARY)
            motion_contours,_=cv2.findContours(motion_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


            if filter_mode == 'balls':
                lbl = "Balls"
                # --- Ball Detection: Color Segmentation ---
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color_combined_mask = None
                for color_name, (lower, upper) in COLOR_RANGES.items():
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    if color_combined_mask is None: color_combined_mask = mask
                    else: color_combined_mask = cv2.bitwise_or(color_combined_mask, mask)

                color_contours = []
                if color_combined_mask is not None:
                    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, COLOR_MASK_CLOSE_KERNEL_SIZE)
                    cleaned_color = cv2.morphologyEx(color_combined_mask, cv2.MORPH_CLOSE, k_close, iterations=1)
                    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, COLOR_MASK_OPEN_KERNEL_SIZE)
                    cleaned_color = cv2.morphologyEx(cleaned_color, cv2.MORPH_OPEN, k_open, iterations=1)
                    thresh_mask_for_display = cleaned_color # Show color mask
                    color_contours, _ = cv2.findContours(cleaned_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours found via COLOR
                color_ball_contours = filter_contours_for_balls(color_contours, f_size_hw)
                color_ball_dets = [cv2.boundingRect(c) for c in color_ball_contours]

                # Filter contours found via MOTION
                motion_ball_contours = filter_contours_for_balls(motion_contours, f_size_hw)
                motion_ball_dets = [cv2.boundingRect(c) for c in motion_ball_contours]

                # Merge detections from both sources (preferring color detections if overlap)
                merged_ball_dets = merge_detections(color_ball_dets, motion_ball_dets)
                # Convert list of tuples back to numpy array for the tracker
                dets = np.array(merged_ball_dets) if merged_ball_dets else np.empty((0,4))


            else: # 'all' mode
                lbl = "Objects"
                thresh_mask_for_display = motion_thresh # Show motion mask
                # Filter contours found via MOTION using general criteria
                filt_contours = filter_contours_adaptive(motion_contours, f_size_hw)
                dets=np.array([cv2.boundingRect(c) for c in filt_contours]) if filt_contours else np.empty((0,4))

            # --- Tracking ---
            tracked=tracker.update(dets,frame,fps)

            # --- Visualization ---
            vis=visualize_tracking(frame,tracked,trajectories,f_count,label=lbl)
            prev_frame=frame.copy()

            # Periodic progress print
            if f_count%50==0:
                elap=time.time()-start_time; avg_f=f_count/elap if elap>0 else 0
                inst_f=1.0/(time.time()-loop_start) if (time.time()-loop_start)>0 else 0
                print(f"(Progress: Frame {f_count}, Avg FPS: {avg_f:.1f}, Inst FPS: {inst_f:.1f})", end='\r')

        # --- Display Frame ---
        if display:
            display_frame=vis if not paused else frame; title=f"Tracking - Mode:{filter_mode}{' (Paused)' if paused else ''}"
            cv2.imshow(title,display_frame)
            mask_window_name="Detection Mask";
            if show_mask:
                if thresh_mask_for_display is not None: cv2.imshow(mask_window_name,thresh_mask_for_display)
                else:
                    try:
                        if cv2.getWindowProperty(mask_window_name,cv2.WND_PROP_VISIBLE)>=1:
                            cv2.imshow(mask_window_name,np.zeros((h//2,w//2),dtype=np.uint8))
                    except: pass
            key=cv2.waitKey(1)&0xFF; quit_pressed=(key==ord('q') or key==27); pause_pressed=(key==ord('p'))
            if quit_pressed: print(f"\nQuit key pressed. Exiting."); break
            if pause_pressed: paused=not paused; print("\nPaused." if paused else "\nResumed.")

        # --- Save Frame ---
        if out_writer and not paused: out_writer.write(vis)

    # --- Cleanup ---
    cap.release()
    if out_writer: out_writer.release() # Use the correct variable name
    if display: cv2.destroyAllWindows()
    processed_frame_count=f_count-(1 if ret else 0);
    if max_frames is not None: processed_frame_count=min(processed_frame_count,max_frames)
    elap_tot=time.time()-start_time; avg_f_fin=processed_frame_count/elap_tot if elap_tot>0 and processed_frame_count>0 else 0
    print("\n"+"-"*25); print(f"Finished processing {processed_frame_count} frames."); print(f"Total Time: {elap_tot:.2f}s"); print(f"Average FPS: {avg_f_fin:.2f}");
    if output_path: print(f"Output saved to: {output_path}")
    print("-"*25)

# --- Main Execution Block ---
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Multi-Object Tracker with Hybrid Ball Detection.")
    parser.add_argument("-v","--video",type=str,required=True,help="Path to video or camera index (e.g., '0').")
    parser.add_argument("-m","--mode",type=str,default="all",choices=["all","balls"],help="Detection mode: 'all'(motion) or 'balls'(hybrid) (default: all).")
    parser.add_argument("-o","--output",type=str,default=None,help="Optional path to save output video (e.g., output/result.mp4).")
    parser.add_argument("--no_display",action="store_true",help="Disable display window.")
    parser.add_argument("--max_frames",type=int,default=None,help="Optional: Process only first N frames.")
    parser.add_argument("--show_mask",action="store_true",help="Show the detection mask (color or motion) in a separate window.")
    args=parser.parse_args()
    process_video(video_path=args.video,filter_mode=args.mode,output_path=args.output,display=not args.no_display,max_frames=args.max_frames, show_mask=args.show_mask)
