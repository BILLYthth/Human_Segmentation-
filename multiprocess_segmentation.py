import cv2
import time
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import subprocess
import psutil
import os
import signal

# --- CẤU HÌNH ---
MODEL_PATH = "../models/unet_int8.tflite"
CAM_W, CAM_H = 320, 240 

def camera_worker(frame_queue, stop_event):
    """Tiến trình lấy ảnh từ Camera"""
    cmd = [
        'rpicam-vid', '-t', '0', '--width', str(CAM_W), '--height', str(CAM_H),
        '--inline', '--nopreview', '--codec', 'yuv420', '--framerate', '30', '-o', '-' 
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = int(CAM_W * CAM_H * 1.5)
    
    try:
        while not stop_event.is_set():
            raw_frame = proc.stdout.read(frame_size)
            if len(raw_frame) != frame_size: continue
            
            yuv = np.frombuffer(raw_frame, dtype=np.uint8).reshape((CAM_H * 3 // 2, CAM_W))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_queue.put(frame)
    finally:
        proc.terminate()
        proc.wait()

def inference_worker(frame_queue, result_queue, stop_event):
    """Tiến trình chạy AI U-Net"""
    # Khởi tạo model bên trong process để tận dụng đa nhân
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    in_scale, in_zero_point = input_details['quantization']
    out_scale, out_zero_point = output_details['quantization']
    dtype = input_details['dtype']

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            t_start = time.time()

            # Pre-processing
            img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # Chuẩn hóa nhanh
            input_data = (img_rgb / 255.0 / in_scale + in_zero_point) if in_scale != 0 else img_rgb
            input_data = np.expand_dims(input_data, axis=0).astype(dtype)

            # Predict
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()

            # Post-processing
            mask_raw = interpreter.get_tensor(output_details['index'])[0]
            mask_prob = (mask_raw.astype(np.float32) - out_zero_point) * out_scale if out_scale != 0 else mask_raw/255.0
            mask = (mask_prob > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST)

            # Overlay nhanh (không dùng copy/maximum tốn tài nguyên)
            overlay = frame.copy()
            overlay[:, :, 1] = np.where(mask > 0, 255, overlay[:, :, 1])
            
            latency_ms = (time.time() - t_start) * 1000

            if result_queue.full():
                try: result_queue.get_nowait()
                except: pass
            result_queue.put((overlay, latency_ms))
        except: continue

def display_worker(result_queue, stop_event):
    """Tiến trình hiển thị và tính toán thống kê"""
    fps_history, cpu_history, lat_history = [], [], []
    fps_count = 0
    last_latency = 0.0
    
    psutil.cpu_percent(interval=None) # Mồi CPU
    last_time = time.time()
    start_run = time.time()

    cv2.namedWindow("Multiprocess Final", cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            try:
                frame, last_latency = result_queue.get(timeout=0.1)
                cv2.imshow("Multiprocess Final", frame)
                fps_count += 1
                lat_history.append(last_latency)
            except: pass

            # Tính toán thông số mỗi giây
            if time.time() - last_time >= 1.0:
                current_cpu = psutil.cpu_percent()
                fps_history.append(fps_count)
                cpu_history.append(current_cpu)
                
                print(f"FPS: {fps_count} | CPU: {current_cpu}% | Latency: {last_latency:.2f}ms")
                fps_count = 0
                last_time = time.time()

            if cv2.waitKey(1) == 27: # Esc để thoát
                stop_event.set()
                break
    finally:
        cv2.destroyAllWindows()
        # In tổng kết - Phần này sẽ chạy kể cả khi p3 bị terminate nhẹ nhàng
        if len(fps_history) > 0:
            print("\n" + "="*50)
            print("📊 TỔNG KẾT HIỆU NĂNG MULTIPROCESS")
            print(f"✔️ FPS trung bình      : {sum(fps_history)/len(fps_history):.2f}")
            print(f"✔️ CPU Usage trung bình : {sum(cpu_history)/len(cpu_history):.2f}%")
            print(f"✔️ Latency trung bình  : {sum(lat_h)/len(lat_h) if (lat_h:=lat_history) else 0:.2f} ms")
            print(f"✔️ Thời gian chạy thực : {time.time() - start_run:.2f} giây")
            print("="*50)

if __name__ == "__main__":
    # Sử dụng Event để đồng bộ việc dừng các process
    stop_event = mp.Event()
    frame_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)
    
    p1 = mp.Process(target=camera_worker, args=(frame_queue, stop_event))
    p2 = mp.Process(target=inference_worker, args=(frame_queue, result_queue, stop_event))
    p3 = mp.Process(target=display_worker, args=(result_queue, stop_event))
    
    p1.start(); p2.start(); p3.start()

    try:
        # Đợi display_worker kết thúc (người dùng nhấn Esc hoặc Ctrl+C)
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Hệ thống] Đang dừng...")
        stop_event.set()
    finally:
        # Để p3 tự in kết quả trước khi tắt hẳn
        p1.terminate()
        p2.terminate()
        p1.join(); p2.join()
        p3.join(timeout=2)
        if p3.is_alive(): p3.terminate()
        print("[Hệ thống] Đã dọn dẹp xong.")