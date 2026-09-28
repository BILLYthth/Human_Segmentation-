import cv2
import time
import numpy as np
import tensorflow as tf
import subprocess
import psutil
import os

# --- CẤU HÌNH ---
MODEL_PATH = "../models/unet_int8.tflite" 
CAM_W, CAM_H = 320, 240         # Độ phân giải tối ưu cho Pi 4
NUM_CORES = psutil.cpu_count()  # Lấy số nhân để tính CPU hệ thống

# 1. Khởi tạo Interpreter (Tối ưu 4 nhân CPU)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_type = input_details['dtype']

# Lấy thông số Quantization để xử lý chính xác model INT8
in_scale, in_zero_point = input_details['quantization']
out_scale, out_zero_point = output_details['quantization']

# 2. Mở luồng Camera rpicam-vid qua Pipe
cmd = [
    'rpicam-vid', '-t', '0', 
    '--width', str(CAM_W), '--height', str(CAM_H),
    '--inline', '--nopreview', '--codec', 'yuv420', 
    '--framerate', '30', 
    '-o', '-' 
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
frame_size = int(CAM_W * CAM_H * 1.5)

print(f"--- Hệ thống Sẵn sàng | Model: {input_type} | Mode: Smooth Optimized ---")

# --- KHỞI TẠO BIẾN ĐO LƯỜNG ---
frame_count = 0
fps_list = []
cpu_list = []
# --- CHỈ BỔ SUNG BIẾN LATENCY ---
latency_list = [] 

# Lấy đối tượng tiến trình hiện tại và tiến trình camera để theo dõi CPU
main_proc = psutil.Process(os.getpid())
main_proc.cpu_percent(interval=None) 
try:
    cam_proc = psutil.Process(proc.pid)
    cam_proc.cpu_percent(interval=None)
except psutil.NoSuchProcess:
    cam_proc = None

global_start_time = time.time()
start_time = time.time()

try:
    while True:
        # Đọc dữ liệu thô
        raw_frame = proc.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            continue
            
        yuv = np.frombuffer(raw_frame, dtype=np.uint8).reshape((CAM_H * 3 // 2, CAM_W))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        # --- BẮT ĐẦU ĐO LATENCY TẠI ĐÂY ---
        t_start = time.time()

        # --- TIỀN XỬ LÝ (GIỮ NGUYÊN) ---
        img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        if in_scale != 0.0:
            normalized_img = img_rgb.astype(np.float32) / 255.0
            x = (normalized_img / in_scale) + in_zero_point
            x = np.clip(x, -128, 127).astype(np.int8) if input_type == np.int8 else np.clip(x, 0, 255).astype(np.uint8)
        else:
            x = img_rgb.astype(input_type)
            if input_type == np.float32: x /= 255.0
            
        x = np.expand_dims(x, axis=0)

        # --- CHẠY AI (GIỮ NGUYÊN) ---
        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()

        # --- HẬU XỬ LÝ (GIỮ NGUYÊN) ---
        mask_raw = interpreter.get_tensor(output_details['index'])[0]
        
        if out_scale != 0.0:
            mask_prob = (mask_raw.astype(np.float32) - out_zero_point) * out_scale
        else:
            mask_prob = mask_raw.astype(np.float32) / 255.0

        mask_prob = cv2.resize(mask_prob, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
        mask_prob = cv2.GaussianBlur(mask_prob, (7, 7), 0)
        mask = (mask_prob > 0.5).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # --- KẾT THÚC ĐO LATENCY TẠI ĐÂY ---
        latency_list.append((time.time() - t_start) * 1000)

        # --- HIỂN THỊ (GIỮ NGUYÊN) ---
        colored_mask = np.zeros_like(frame)
        colored_mask[:, :, 1] = mask 
        overlay = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

        cv2.imshow("Optimized U-Net Smooth", overlay)

        # --- ĐO LƯỜNG VÀ HIỂN THỊ (GIỮ NGUYÊN LOGIC CŨ) ---
        frame_count += 1
        if time.time() - start_time >= 1.0:
            total_cpu = main_proc.cpu_percent(interval=None)
            if cam_proc:
                try:
                    total_cpu += cam_proc.cpu_percent(interval=None)
                except psutil.NoSuchProcess:
                    pass
            
            # Chuẩn hóa về 100% (Theo yêu cầu của bạn)
            current_cpu = total_cpu / NUM_CORES if NUM_CORES else total_cpu
            
            fps_list.append(frame_count)
            cpu_list.append(current_cpu)
            
            # Bổ sung in Latency tức thời vào console
            print(f"FPS: {frame_count} | CPU Usage: {current_cpu:.2f}% | Latency: {latency_list[-1]:.2f}ms")
            
            frame_count = 0
            start_time = time.time()

        if cv2.waitKey(1) == 27: break 

except KeyboardInterrupt: pass
except Exception as e: print(f"Lỗi vận hành: {e}")

finally:
    proc.kill()
    cv2.destroyAllWindows()
    
    # --- TỔNG KẾT (CHỈ THÊM DÒNG LATENCY TRUNG BÌNH) ---
    print("\n" + "="*50)
    print("📊 TỔNG KẾT HIỆU NĂNG SINGLE PROCESS")
    print("="*50)
    
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
    avg_cpu = sum(cpu_list) / len(cpu_list) if cpu_list else 0.0
    avg_latency = sum(latency_list) / len(latency_list) if latency_list else 0.0
    total_time = time.time() - global_start_time
    
    print(f"✔️ Thời gian chạy thực tế : {total_time:.2f} giây")
    print(f"✔️ FPS trung bình         : {avg_fps:.2f}")
    print(f"✔️ CPU Usage trung bình   : {avg_cpu:.2f}%")
    print(f"✔️ Latency trung bình     : {avg_latency:.2f} ms")
    print("="*50)