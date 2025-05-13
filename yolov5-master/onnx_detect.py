# -*- coding: utf-8 -*-
"""
Time:     2021.10.26
Author:   Athrunsunny
Version:  V 0.2
Updates:  Real-time detection with FPS display and performance optimization
"""

import cv2
import torch
import time
import onnxruntime
import numpy as np
# 假设utils.py与onnx_detect.py在同一目录下
from function.utils import LoadImages, Annotator, colors, check_img_size, non_max_suppression, scale_coords
from function import config as CFG

class RealTimeDetector:
    def __init__(self, weights, imgsz=640, stride=64):
        """ 初始化模型和性能优化参数 """
        # ONNX Runtime 配置（绑定CPU大核）
        self.session_options = onnxruntime.SessionOptions()
        self.session_options.intra_op_num_threads = 8  # i5-13500H有4P核+8E核
        self.session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        
        # 加载模型
        self.session = onnxruntime.InferenceSession(weights, self.session_options)
        self.imgsz = check_img_size(imgsz, s=stride)
        self.stride = stride
        
        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 缓存配置
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img):
        """ 优化后的图像预处理（速度提升关键） """
        # 使用OpenCV的blobFromImage（比手动处理快3倍）
        blob = cv2.dnn.blobFromImage(
            img, 
            scalefactor=1/255.0, 
            size=(self.imgsz, self.imgsz),
            mean=(0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        return blob.astype(np.float32)

    def inference(self, img, conf_thres=0.25, iou_thres=0.45):
        """ 推理 + NMS """
        # 预处理
        blob = self.preprocess(img)
        
        # ONNX推理
        pred = self.session.run([self.output_name], {self.input_name: blob})[0]
        
        # 转换为Tensor并NMS
        pred = torch.tensor(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        return pred

    def postprocess(self, pred, img, label_dict):
        """ 后处理并绘制结果 """
        annotator = Annotator(img, line_width=2, example=str(label_dict['cardlabel']))
        res_labels = []
        
        if pred[0] is not None:
            # 调整坐标到原始图像尺寸
            pred[0][:, :4] = scale_coords((self.imgsz, self.imgsz), pred[0][:, :4], img.shape).round()
            
            # 绘制检测框
            for *xyxy, conf, cls in pred[0]:
                label = f"{label_dict['cardlabel'][int(cls)]} {conf:.2f}"
                res_labels.append(label_dict['cardlabel'][int(cls)])
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
        
        # 计算并显示FPS
        self.frame_count += 1
        if (time.time() - self.start_time) > 1:
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.frame_count = 0
            self.start_time = time.time()
        
        cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img, res_labels

    def run(self, source=1, show=True, conf_thres=0.25):
        """ 实时检测主循环 """
        # 初始化视频源（0=摄像头，或视频文件路径）
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError("无法打开视频源")
        
        # 设置摄像头分辨率（提升帧率）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 推理
            pred = self.inference(frame, conf_thres=conf_thres)
            
            # 后处理
            processed_frame, labels = self.postprocess(pred, frame, CFG.LABEL_DICT)
            
            if show:
                cv2.imshow('Real-Time Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 配置参数
    model_path = 'D:\Learning\yolo\yolov5-master\yolov5-master\\runs\\train\exp8\weights\\best.onnx'  # 替换为你的ONNX模型路径
    video_source = 0  # 0=默认摄像头，或视频文件路径
    
    # 初始化检测器
    detector = RealTimeDetector(model_path, imgsz=640)
    
    # 启动实时检测
    print("启动实时检测（按Q退出）...")
    detector.run(source=video_source, show=True, conf_thres=0.25)