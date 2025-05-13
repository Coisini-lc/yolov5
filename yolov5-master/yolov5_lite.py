import cv2
import numpy as np
import argparse
import onnxruntime as ort
import time

class yolov5_lite():
    def __init__(self, model_pb_path, label_path, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        """
        YOLOv5 Lite模型初始化
        :param model_pb_path: ONNX模型路径
        :param label_path: 类别标签文件路径
        :param confThreshold: 类别置信度阈值
        :param nmsThreshold: 非极大值抑制阈值
        :param objThreshold: 目标存在置信度阈值
        """
        # 配置ONNX Runtime会话选项（关闭冗余日志）
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)  # 加载ONNX模型
        
        # 读取类别标签文件
        self.classes = list(map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)  # 类别数量
        
        # YOLO锚点配置（不同尺度的先验框）
        anchors = [[10, 13, 16, 30, 33, 23], 
                  [30, 61, 62, 45, 59, 119], 
                  [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)  # 检测层数量
        self.na = len(anchors[0]) // 2  # 每个检测层的锚点数量
        self.no = self.num_classes + 5   # 每个预测框的输出维度（类别数 + 坐标4 + 置信度1）
        
        # 初始化网格和步长参数
        self.grid = [np.zeros(1)] * self.nl  # 网格缓存
        self.stride = np.array([8., 16., 32.])  # 特征图相对于输入图像的缩放步长
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)  # 锚点网格
        
        # 阈值参数
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        
        # 输入图像尺寸（从模型元数据获取）
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])

    # 调整输入图像尺寸并保持宽高比（必要时添加黑边）
    # return: 调整后的图像, 新尺寸, 填充信息
    # 图像 240 320 40 0
    def resize_image(self, srcimg, keep_ratio=True):
        """
        调整输入图像尺寸并保持宽高比（必要时添加黑边）
        :param srcimg: 原始图像
        :param keep_ratio: 是否保持宽高比
        :return: 调整后的图像, 新尺寸, 填充信息
        """
        # 先按输入图像等比例缩小，然后再对小的部分进行填充
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:  # 高 > 宽
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)  # 计算左右填充
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, 
                                        cv2.BORDER_CONSTANT, value=0)  # 添加黑边
            else:  # 宽 > 高
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)  # 计算上下填充
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)  # 直接缩放
        return img, newh, neww, top, left  # 返回调整后的图像和填充信息

    # 生成网格坐标，用于调整预测框的绝对位置
    def _make_grid(self, nx=20, ny=20):
        """生成网格坐标，用于调整预测框的绝对位置"""
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))          #按行以及按列复制相应大小的矩阵
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)        # 生成网格坐标
    
    # 解析模型输出，筛选检测框，应用NMS，绘制结果
    # return: 绘制检测框后的图像
    def postprocess(self, frame, outs, pad_hw):
        #找到框的坐标，以及什么类别，对框进行非极大值抑制
        """
        后处理:解析模型输出,筛选检测框,应用NMS,绘制结果
        :param frame: 原始图像
        :param outs: 模型输出（未处理的预测框）
        :param pad_hw: 调整图像时的填充信息 (newh, neww, padh, padw)
        :return: 绘制检测框后的图像
        """
        newh, neww, padh, padw = pad_hw # 240 320 40 0
        frameHeight, frameWidth = frame.shape[0], frame.shape[1] # 480 640
        ratioh, ratiow = frameHeight / newh, frameWidth / neww  # 尺寸还原比例
        
        classIds, confidences, boxes = [], [], []
        for detection in outs:
            scores = detection[5:]  # 类别置信度，提取类别分数（索引5之后）
            classId = np.argmax(scores) # 找到最高置信度的类别ID
            confidence = scores[classId] # 获取该类别置信度
            # 双重阈值筛选：类别置信度 + 目标存在置信度
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                # x中心点 y中心点 宽度 高度
                # [166.28496 220.06169 126.179634 107.74112 0.50108314 0.9869262 ]
                # 将相对坐标转换为原始图像坐标
                center_x = int((detection[0] - padw) * ratiow)
                center_y = int((detection[1] - padh) * ratioh)
                # width 和 height 为检测框在原始图像中的实际宽度和高度
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # 保存结果
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # 非极大值抑制（NMS）去除重叠框
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0] if isinstance(i, (tuple, list)) else i  # 兼容不同OpenCV版本
            box = boxes[i]
            left, top, width, height = box
            # 绘制检测框和标签
            frame = self.drawPred(frame, classIds[i], confidences[i], 
                                 left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        """在图像上绘制检测框和标签"""
        # 绘制矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)
        # 构造标签文本（类别+置信度）
        label = f"{self.classes[classId]}: {conf:.2f}"
        # 计算文本尺寸并绘制
        # (label_width, label_height), baseline = cv2.getTextSize(
        #     label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        # )
        cv2.putText(frame, label, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    def detect(self, srcimg):
        """完整检测流程：预处理→推理→后处理"""
        # 调整图像尺寸并预处理
        img, newh, neww, top, left = self.resize_image(srcimg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
        img = img.astype(np.float32) / 255.0        # 归一化
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # 调整维度顺序：NCHW
        
        # 模型推理
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        # 处理输出：调整预测框坐标到输入图像尺度
        row_ind = 0
        for i in range(self.nl):
            h = int(self.input_shape[0] / self.stride[i])  # 特征图高度 40
            w = int(self.input_shape[1] / self.stride[i])  # 特征图宽度 40
            length = int(self.na * h * w)  # 当前层的预测框数量
            # 生成网格（如果尺寸变化）
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)
            # 调整坐标（YOLO解码过程）
            outs[row_ind:row_ind+length, 0:2] = (outs[row_ind:row_ind+length, 0:2] * 2. - 0.5 + 
                                               np.tile(self.grid[i], (self.na, 1))) * self.stride[i]
            outs[row_ind:row_ind+length, 2:4] = (outs[row_ind:row_ind+length, 2:4] * 2) ** 2 * \
                                               np.repeat(self.anchor_grid[i], h*w, axis=0)
            row_ind += length
        
        # 后处理并返回结果图像
        return self.postprocess(srcimg, outs, (newh, neww, top, left))


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default=r'D:\\Learning\\yolo\\yolov5-master\\yolov5-master\\runs\\train\\exp8\weights\best.onnx', help="ONNX模型路径")
    parser.add_argument('--classfile', type=str, default=r'D:\\Learning\\yolo\\yolov5-master\\yolov5-master\\VOC\\classes.txt', help="类别文件路径")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='类别置信度阈值')
    parser.add_argument('--nmsThreshold', default=0.6, type=float, help='NMS IOU阈值')
    args = parser.parse_args()

    # 初始化检测器
    net = yolov5_lite(args.modelpath, args.classfile, 
                     confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)

    # 视频流处理（默认摄像头）
    capture = cv2.VideoCapture(1)  # 0表示默认摄像头
    counter, start_time = 0, time.time()
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        # 计算并显示实时FPS
        counter += 1
        if (time.time() - start_time) > 0:
            fps = counter / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            counter, start_time = 0, time.time()  # 重置计数器
        
        # 执行检测并显示结果
        detected_frame = net.detect(frame)
        cv2.imshow("Real-time Detection", detected_frame)
        
        # 按'q'退出
        if cv2.waitKey(20) & 0xff == ord('q'):
            break

    # 释放资源
    capture.release()
    cv2.destroyAllWindows()