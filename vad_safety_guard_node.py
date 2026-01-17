import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
import numpy as np
import csv
import os
from datetime import datetime
from autoware_internal_planning_msgs.msg import CandidateTrajectories
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from nav_msgs.msg import Odometry

class VadSafetyGuard(Node):
    def __init__(self):
        super().__init__('vad_safety_guard')
        
        self.declare_parameter('system_delay', 0.5)
        self.declare_parameter('max_decel', 3.0)
        self.declare_parameter('max_lat_acc', 3.0)

        # QoS 설정
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.sub_candidates = self.create_subscription(CandidateTrajectories, '/planning/vad/candidate_trajectories', self.on_candidates, sub_qos)
        self.sub_odom = self.create_subscription(Odometry, '/localization/kinematic_state', self.on_odom, sub_qos)
        self.pub_trajectory = self.create_publisher(Trajectory, '/planning/trajectory_guarded', pub_qos)

        self.current_velocity = 0.0
        self.current_pose = None
        self.last_odom_time = self.get_clock().now()
        self.last_selected_curvature = 0.0

        # 로그 파일 생성
        self.init_logger()

    def init_logger(self):
        # 파일 저장
        file_name = f"safety_guard_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_path = os.path.join('/workspace', file_name)
        
        try:
            self.csv_file = open(self.log_path, 'w', newline='')
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(["Timestamp", "Velocity", "Selected_Rank", "Curvature", "Lateral_Accel", "Is_Emergency_Stop"])
            self.get_logger().info(f"==== LOG FILE WILL BE SAVED TO: {self.log_path} ====")
        except Exception as e:
            self.get_logger().warn(f"Cannot save to /workspace ({e}). Saving to current directory.")
            self.log_path = file_name
            self.csv_file = open(self.log_path, 'w', newline='')
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(["Timestamp", "Velocity", "Selected_Rank", "Curvature", "Lateral_Accel", "Is_Emergency_Stop"])

    def on_odom(self, msg):
        self.current_velocity = msg.twist.twist.linear.x
        self.current_pose = msg.pose.pose
        self.last_odom_time = self.get_clock().now()

    def on_candidates(self, msg):
        selected_rank = -1
        final_curvature = 0.0
        lat_accel = 0.0
        is_emergency = 0

        # Odom Timeout 체크
        if (self.get_clock().now() - self.last_odom_time).nanoseconds > 1.0 * 1e9:
            self.publish_stop(msg)
            return

        delay = self.get_parameter('system_delay').value
        a_max = self.get_parameter('max_decel').value
        lat_max = self.get_parameter('max_lat_acc').value
        
        sorted_trajs = msg.candidate_trajectories
        feasible_trajs = [] 

        for i, traj in enumerate(sorted_trajs):
            if self.check_feasibility(traj, self.current_velocity, delay, a_max, lat_max):
                avg_curv = self.get_avg_curvature(traj)
                feasible_trajs.append({'original_idx': i, 'curvature': avg_curv, 'trajectory': traj})

        output_header = Header()
        output_header.stamp = self.get_clock().now().to_msg()
        if len(msg.candidate_trajectories) > 0:
            output_header.frame_id = msg.candidate_trajectories[0].header.frame_id
        else:
            output_header.frame_id = "map"

        output = Trajectory()
        output.header = output_header
        
        if len(feasible_trajs) > 0:
            # 이전 곡률과 가장 비슷한 것 선택
            best_candidate = min(
                feasible_trajs, 
                key=lambda x: abs(x['curvature'] - self.last_selected_curvature)
            )
            output.points = best_candidate['trajectory'].points
            self.last_selected_curvature = best_candidate['curvature']
            
            selected_rank = best_candidate['original_idx']
            final_curvature = best_candidate['curvature']
            lat_accel = (self.current_velocity ** 2) * abs(final_curvature)
            
            if selected_rank > 0:
                self.get_logger().info(f"Guard Intervened: Rank {selected_rank}")
        else:
            output = self.generate_emergency_stop_path(output_header)
            self.last_selected_curvature = 0.0
            is_emergency = 1

        self.pub_trajectory.publish(output)
        
        # [데이터 기록]
        try:
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.writer.writerow([current_time, self.current_velocity, selected_rank, final_curvature, lat_accel, is_emergency])
        except:
            pass

    def check_feasibility(self, traj, v_curr, t_delay, a_max, lat_max):
        points = traj.points
        if len(points) < 5: return False 
        try:
            p_start = np.array([points[0].pose.position.x, points[0].pose.position.y])
            p_end = np.array([points[-1].pose.position.x, points[-1].pose.position.y])
            traj_len = np.linalg.norm(p_end - p_start)
            req_dist = (v_curr * t_delay) + (v_curr**2) / (2 * a_max)
            if points[-1].longitudinal_velocity_mps < 0.1 and traj_len < req_dist: return False

            idx_mid = len(points) // 2
            p1 = np.array([points[0].pose.position.x, points[0].pose.position.y])
            p2 = np.array([points[idx_mid].pose.position.x, points[idx_mid].pose.position.y])
            p3 = np.array([points[-1].pose.position.x, points[-1].pose.position.y])
            curvature = self.calculate_curvature(p1, p2, p3)
            lat_accel = (v_curr ** 2) * curvature
            if lat_accel > lat_max: return False
        except: return False 
        return True

    def get_avg_curvature(self, traj):
        try:
            points = traj.points
            if len(points) < 3: return 0.0
            p1 = np.array([points[0].pose.position.x, points[0].pose.position.y])
            p2 = np.array([points[len(points)//2].pose.position.x, points[len(points)//2].pose.position.y])
            p3 = np.array([points[-1].pose.position.x, points[-1].pose.position.y])
            return self.calculate_curvature(p1, p2, p3)
        except: return 0.0

    def calculate_curvature(self, p1, p2, p3):
        try:
            area = 0.5 * np.abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            d12 = np.linalg.norm(p1 - p2); d23 = np.linalg.norm(p2 - p3); d31 = np.linalg.norm(p3 - p1)
            if d12 * d23 * d31 == 0: return 0.0
            return (4 * area) / (d12 * d23 * d31)
        except: return 0.0

    def publish_stop(self, msg):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        if len(msg.candidate_trajectories) > 0: header.frame_id = msg.candidate_trajectories[0].header.frame_id
        out = self.generate_emergency_stop_path(header)
        self.pub_trajectory.publish(out)

    def generate_emergency_stop_path(self, header):
        traj = Trajectory()
        traj.header = header
        point = TrajectoryPoint()
        if "map" in header.frame_id and self.current_pose: point.pose = self.current_pose
        else: point.pose.position.x = 0.0; point.pose.position.y = 0.0; point.pose.orientation.w = 1.0
        point.longitudinal_velocity_mps = 0.0; point.acceleration_mps2 = -3.0
        traj.points.append(point)
        return traj

def main(args=None):
    rclpy.init(args=args)
    node = VadSafetyGuard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()