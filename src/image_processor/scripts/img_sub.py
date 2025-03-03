import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            Image,
            'camera/image',  # 订阅原始相机话题
            self.img_callback,
            10)
        self.publisher = self.create_publisher(
            Image,
            'processed_image',  # 发布处理后的图像
            10)
        self.bridge = CvBridge()
        self.get_logger().info("Image Processor Node Started")

    def img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image message: {e}")
            return

        # 图像处理：检测红色物体
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # 使用两个范围覆盖鲜红色
        lower_red1 = np.array([0, 150, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 使用形态学操作合并分割的区域
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算：膨胀 + 腐蚀

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到面积最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 2000:  # 提高面积阈值
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 在方框左上角显示坐标
                text = f"({x}, {y})"
                cv2.putText(cv_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.get_logger().info(f"Object detected at: ({x}, {y})")
        
        # 发布处理后的图像
        self.publisher.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Image Processor Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
