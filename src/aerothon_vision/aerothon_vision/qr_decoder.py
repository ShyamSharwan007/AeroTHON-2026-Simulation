import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pyzbar.pyzbar import decode

class QRDecoderNode(Node):
    def __init__(self):
        super().__init__('qr_decoder_node')
        self.subscription = self.create_subscription(
            Image, '/drone/bottom_camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        
        # --- MISSION STATE VARIABLES ---
        self.target_payload_id = None
        self.mission_state = "INITIALIZATION" 
        
        self.get_logger().info('Vision Node Active. State: INITIALIZATION')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return

        decoded_objects = decode(cv_image)

        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            points = obj.polygon
            
            if len(points) == 4:
                pts = np.array([(point.x, point.y) for point in points], dtype=np.int32)
                
                # Calculate the center pixel (cx, cy) of the bounding box
                cx = int(np.mean([p.x for p in points]))
                cy = int(np.mean([p.y for p in points]))

                # --- MISSION LOGIC ---
                if self.mission_state == "INITIALIZATION":
                    self.target_payload_id = qr_data
                    self.mission_state = "SEARCHING"
                    self.get_logger().info(f"START TARGET ACQUIRED! Saved ID: {self.target_payload_id}")
                    
                    # Draw Blue Box for Start Target
                    cv2.polylines(cv_image, [pts], True, (255, 0, 0), 3) 
                    cv2.putText(cv_image, "INITIALIZED", (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                elif self.mission_state == "SEARCHING":
                    if qr_data == self.target_payload_id:
                        # CORRECT DELIVERY TARGET
                        self.get_logger().info(f"MATCH CONFIRMED! Center Pixel: ({cx}, {cy})")
                        cv2.polylines(cv_image, [pts], True, (0, 255, 0), 3) # Green Box
                        cv2.putText(cv_image, "MATCH", (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1) # Red Target Dot
                    else:
                        # DECOY TARGET
                        cv2.polylines(cv_image, [pts], True, (0, 0, 255), 3) # Red Box
                        cv2.putText(cv_image, "DECOY", (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("AeroTHON Drone Vision [Live]", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = QRDecoderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
