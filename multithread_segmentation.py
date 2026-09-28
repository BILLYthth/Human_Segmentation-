import cv2
import threading
import queue
import numpy as np
import tensorflow as tf
import time
import subprocess
import psutil

# --- CẤU HÌNH ---
MODEL_PATH = "../models/unet_int8.tflite"
CAM_W, CAM_H = 320, 240

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def camera_thread():
    # Dùng rpicam-vid cho đúng bài
    cmd = [
        'rpicam-vid', '-t', '0', '--width', str(CAM_W), '--height', str(CAM_H),
        '--inline', '--nopreview', '--codec', 'yuv420', '--framerate', '30', '-o', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = int(CAM_W * CAM_H * 1.5)
    try:
        while True:
            raw_frame = proc.stdout.read(frame_size)
            if len(raw_frame) != frame_size: continue
            yuv = np.frombuffer(raw_frame, dtype=np.uint8).reshape((CAM_H * 3 // 2, CAM_W))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            if not frame_queue.full():
                frame_queue.put(frame)
    finally:
        proc.kill()

def inference_thread():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    in_scale, in_zero_point = input_details['quantization']
    out_scale, out_zero_point = output_details['quantization']
    target_dtype = input_details['dtype']

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            t_start = time.time()

            img = cv2.resize(frame, (256, 256))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            normalized = img_rgb.astype(np.float32) / 255.0
            
            input_data = (normalized / in_scale + in_zero_point) if in_scale != 0 else normalized * 255
            input_data = np.expand_dims(input_data, axis=0).astype(target_dtype)

            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()

            mask_raw = interpreter.get_tensor(output_details['index'])[0]
            mask_prob = (mask_raw.astype(np.float32) - out_zero_point) * out_scale if out_scale != 0 else mask_raw.astype(np.float32) / 255.0
            
            mask = (mask_prob > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (CAM_W, CAM_H))

            overlay = frame.copy()
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], mask)
            
            latency = (time.time() - t_start) * 1000
            result_queue.put((overlay, latency))

def display_thread(stop_event):
    fps_count = 0
    last_print = time.time()
    start_time = time.time()
    
    # Khởi tạo list lưu trữ để tính trung bình
    fps_history = []
    cpu_history = []
    lat_history = []
    
    current_lat = 0.0 # Tránh lỗi UnboundLocalError
    psutil.cpu_percent(interval=None)

    try:
        while not stop_event.is_set():
            if not result_queue.empty():
                frame, current_lat = result_queue.get()
                cv2.imshow("Multithread Lab", frame)
                fps_count += 1
                lat_history.append(current_lat)

            if time.time() - last_print >= 1.0:
                cpu = psutil.cpu_percent()
                fps_history.append(fps_count)
                cpu_history.append(cpu)
                
                print(f"FPS: {fps_count} | CPU: {cpu}% | Latency: {current_lat:.2f}ms")
                fps_count = 0
                last_print = time.time()

            if cv2.waitKey(1) == 27: # Nhấn ESC để thoát
                stop_event.set()
                break
    finally:
        # ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT: Bắt buộc in tổng kết dù bị Ctrl+C
        cv2.destroyAllWindows()
        print("\n" + "="*50)
        print("📊 TỔNG KẾT HIỆU NĂNG MULTITHREAD")
        if fps_history:
            print(f"✔️ FPS trung bình      : {sum(fps_history)/len(fps_history):.2f}")
            print(f"✔️ CPU trung bình      : {sum(cpu_history)/len(cpu_history):.2f}%")
            print(f"✔️ Latency trung bình  : {sum(lat_history)/len(lat_history):.2f} ms")
            print(f"✔️ Tổng thời gian chạy : {time.time() - start_time:.2f} s")
        else:
            print("❌ Không có đủ dữ liệu để tính trung bình.")
        print("="*50)

if __name__ == "__main__":
    stop_event = threading.Event()
    t1 = threading.Thread(target=camera_thread, daemon=True)
    t2 = threading.Thread(target=inference_thread, daemon=True)
    t3 = threading.Thread(target=display_thread, args=(stop_event,))
    
    t1.start(); t2.start(); t3.start()
    
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Hệ thống] Đang dừng...")
        stop_event.set()
    
    t3.join()
    print("[Hệ thống] Kết thúc.")