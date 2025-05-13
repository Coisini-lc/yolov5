import cv2

# 检查DNN模块是否可用
print("=== OpenCV DNN支持情况 ===")
print("OpenCV版本:", cv2.__version__)
print("DNN模块可用:", hasattr(cv2, 'dnn'))

if hasattr(cv2, 'dnn'):
    # 获取所有支持的后端和计算目标
    backends = {
        cv2.dnn.DNN_BACKEND_OPENCV: "OPENCV",
        cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE: "INTEL",
        cv2.dnn.DNN_BACKEND_CUDA: "CUDA"
    }
    
    print("\n可用计算目标：")
    for backend_id, backend_name in backends.items():
        try:
            targets = cv2.dnn.getAvailableTargets(backend_id)
            print(f"{backend_name}: {[cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD, cv2.dnn.DNN_TARGET_VULKAN, cv2.dnn.DNN_TARGET_FPGA, cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16] if targets else '不支持'}")
        except:
            print(f"{backend_name}: 检测失败")