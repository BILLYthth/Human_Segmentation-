import subprocess

print("Running Single Process")
subprocess.run(["python3","single_process_segmentation.py"])

print("Running Multithread Version")
subprocess.run(["python3","multithread_segmentation.py"])

print("Running Multiprocess Version")
subprocess.run(["python3","multiprocess_segmentation.py"])

