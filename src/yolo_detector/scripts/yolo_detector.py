#!/usr/bin/env python3
import rclpy  # ROS2 Python API
from rclpy.node import Node
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String  # 用于发布检测结果

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # 创建 ROS 2 发布者，发布检测到的物体信息
        self.publisher = self.create_publisher(String, 'detected_object', 10)

        # 订阅摄像头图像话题（更符合 ROS 2 规范）
        self.subscription = self.create_subscription(
            Image, 
            '/camera/image',  # 需要确保摄像头节点发布该话题
            self.image_callback, 
            10)
        
        self.bridge = CvBridge()  # 用于 ROS2 与 OpenCV 图像转换
        self.model = YOLO('yolov8n.pt')  # 加载 YOLOv8 模型
        self.get_logger().info("YOLO Detector Node started!")

    def image_callback(self, msg):
        """ 处理摄像头图像 """
        try:
            # ROS 图像转换为 OpenCV 格式
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 运行 YOLOv8 目标检测
            results = self.model(frame)

            # 遍历检测结果
            for result in results:
                boxes = result.boxes.cpu().numpy()  # 获取检测框
                for box in boxes:
                    xmin, ymin, xmax, ymax = box.xyxy[0]  # 获取检测框坐标
                    confidence = box.conf[0]  # 置信度
                    class_id = int(box.cls[0])  # 类别 ID
                    class_name = self.model.names[class_id]  # 类别名称

                    if confidence > 0.5:  # 置信度阈值
                        self.get_logger().info(f"Detected: {class_name} at ({xmin}, {ymin}, {xmax}, {ymax})")

                        # 发布检测结果
                        msg = String()
                        msg.data = f"{class_name}: ({xmin}, {ymin}, {xmax}, {ymax})"
                        self.publisher.publish(msg)

            # 显示检测结果（可选）
            annotated_frame = results[0].plot()  # 绘制检测框
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    try:
        rclpy.spin(node)  # ROS2 运行
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO Detector Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
