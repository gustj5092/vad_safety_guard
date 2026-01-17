import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from autoware_control_msgs.msg import Control
from collections import deque
import time

class CmdDelayer(Node):
    def __init__(self):
        super().__init__('cmd_delayer')
        self.declare_parameter('delay_sec', 0.0)
        self.queue = deque()
        
        # QoS 설정 (Volatile / Transient Local)
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.sub = self.create_subscription(Control, '/control/command/control_cmd', self.msg_callback, sub_qos)
        self.pub = self.create_publisher(Control, '/control/command/control_cmd_delayed', pub_qos)

    def msg_callback(self, msg):
        # 콜백마다 파라미터 확인 (실시간 변경 가능)
        current_delay = self.get_parameter('delay_sec').value
        now = time.time()
        self.queue.append((now, msg))
        
        while self.queue:
            timestamp, stored_msg = self.queue[0]
            if now - timestamp >= current_delay:
                current_ros_time = self.get_clock().now().to_msg()
                stored_msg.stamp = current_ros_time 
                self.pub.publish(stored_msg)
                self.queue.popleft()
            else:
                break

def main():
    rclpy.init()
    node = CmdDelayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()