#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('img_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image', 10)  # 创建发布者
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        self.timer = self.create_timer(0.1, self.publish_image)  # 设定定时器，每0.1秒调用一次

    def publish_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.get_logger().info('Publishing image')
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")  # OpenCV图像转换为ROS2的Image消息
            self.publisher_.publish(img_msg)  # 发布消息
        else:
            self.get_logger().warn('Failed to capture image')


def main(args=None):
    rclpy.init(args=args)  # 初始化rclpy
    node = ImagePublisher()  # 创建节点对象
    try:
        rclpy.spin(node)  # 运行ROS2节点，直到手动关闭
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()  # 释放摄像头
        node.destroy_node()  # 销毁节点
        rclpy.shutdown()  # 关闭rclpy


if __name__ == '__main__':
    main()
