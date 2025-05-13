import cv2
import numpy as np
import time
import os
import onnxruntime as ort

os.environ['OPENCV_DNN_CPU_RUNTIME'] = 'OMP'  # 强制OpenMP
os.environ['OMP_NUM_THREADS'] = '8'           # 匹配物理核心数
os.environ['KMP_BLOCKTIME'] = '1'             # Intel优化参数
class YOLOv5Detector:
    def __init__(self, model_path, class_path, is_cuda=False):
        # 参数设置
        self.INPUT_WIDTH = 640.0
        self.INPUT_HEIGHT = 640.0
        self.SCORE_THRESHOLD = 0.4
        self.NMS_THRESHOLD = 0.4
        self.CONFIDENCE_THRESHOLD = 0.4
        
        # 加载网络（强制使用CPU）
        self.net = cv2.dnn.readNet(model_path)
        sess = ort.InferenceSession(model_path)
        print("输入形状:", sess.get_inputs()[0].shape)  # 应为[1,3,640,640] 
        # 取消注释并确保以下代码执行
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 必须取消注释
        self.net.enableWinograd(False)  # CPU上必须关闭
        # 加载类别名称
        with open(class_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    
    def format_yolov5(self, source):
        # 图像预处理：保持长宽比并填充
        col, row = source.shape[1], source.shape[0]
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), dtype=np.uint8)
        result[0:row, 0:col] = source
        return result
    
    def detect(self, image):
        # 预处理
        input_image = self.format_yolov5(image)
        blob = cv2.dnn.blobFromImage(input_image, 1/255.0, 
                                    (int(self.INPUT_WIDTH), int(self.INPUT_HEIGHT)), 
                                    swapRB=True, crop=False)
        
        # 设置输入并前向传播
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        # 计算缩放因子
        x_factor = input_image.shape[1] / self.INPUT_WIDTH
        y_factor = input_image.shape[0] / self.INPUT_HEIGHT
        
        # 解析输出
        class_ids = []
        confidences = []
        boxes = []
        
        # YOLOv5输出格式为(1, 25200, 85)
        output = outputs[0][0]
        
        for detection in output:
            confidence = detection[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:
                scores = detection[5:]
                class_id = np.argmax(scores)
                max_score = scores[class_id]
                
                if max_score > self.SCORE_THRESHOLD:
                    # 转换坐标
                    x, y, w, h = detection[0], detection[1], detection[2], detection[3]
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # NMS处理
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                  self.CONFIDENCE_THRESHOLD, 
                                  self.NMS_THRESHOLD)
        
        # 整理结果
        results = []
        if isinstance(indices, np.ndarray):  # 确保indices是numpy数组
            for i in indices.flatten():
                results.append({
                    'class_id': class_ids[i],
                    'confidence': confidences[i],
                    'box': boxes[i]
                })
        return results
    
    def draw_prediction(self, image, results):
        # 绘制检测结果
        for detection in results:
            box = detection['box']
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # 随机颜色
            color = (np.random.randint(0, 256), 
                     np.random.randint(0, 256), 
                     np.random.randint(0, 256))
            
            # 绘制边界框
            cv2.rectangle(image, 
                         (box[0], box[1]), 
                         (box[0] + box[2], box[1] + box[3]), 
                         color, 2)
            
            # 绘制标签背景
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(image, 
                         (box[0], box[1] - 20), 
                         (box[0] + len(label) * 10, box[1]), 
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(image, label, 
                       (box[0], box[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 0, 0), 1)

def main():
    
    # 参数设置
    model_path = "D:/Learning/yolo/yolov5-master/yolov5-master/runs/train/exp8/weights/best.onnx"  # 使用正斜杠避免转义
    class_path = "class.names"
    video_path = 1  # 或使用0表示摄像头
    
    # 初始化检测器（强制使用CPU）
    detector = YOLOv5Detector(model_path, class_path, is_cuda=False)  # 显式设置为False
    
    # 打开视频源
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频源")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        start_time = time.time()
        results = detector.detect(frame)
        fps = 1.0 / (time.time() - start_time)
        
        # 绘制结果
        detector.draw_prediction(frame, results)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("YOLOv5 Detection (CPU)", frame)  # 修改窗口标题
        
        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()