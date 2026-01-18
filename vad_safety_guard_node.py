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

# [핵심] CARLA 충돌 메시지 타입 임포트
# 만약 "No module named 'carla_msgs'" 에러가 나면, 맨 아래 '주의사항'을 봐주세요.
try:
    from carla_msgs.msg import CarlaCollisionEvent
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

class VadSafetyGuard(Node):
    def __init__(self):
        super().__init__('vad_safety_guard')
        
        self.declare_parameter('system_delay', 0.5)  
        self.declare_parameter('max_decel', 3.0)  
        self.declare_parameter('max_lat_acc', 3.0)  
        self.declare_parameter('selection_strategy', 'follow_last_curvature') 

        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)

        # 1. 기존 구독자들
        self.sub_candidates = self.create_subscription(CandidateTrajectories, '/planning/vad/candidate_trajectories', self.on_candidates, sub_qos)
        self.sub_unchecked = self.create_subscription(Trajectory, '/planning/vad/unchecked_trajectory', self.on_unchecked, sub_qos)
        self.sub_odom = self.create_subscription(Odometry, '/localization/kinematic_state', self.on_odom, sub_qos)
        
        # [NEW] 2. CARLA 충돌 토픽 구독
        if CARLA_AVAILABLE:
            self.sub_collision = self.create_subscription(CarlaCollisionEvent, '/carla/ego_vehicle/collision', self.on_collision, sub_qos)
            self.get_logger().info("✅ CARLA Collision Sensor Subscribed!")
        else:
            self.get_logger().warn("⚠️ carla_msgs not found! Collision detection disabled.")

        self.pub_trajectory = self.create_publisher(Trajectory, '/planning/trajectory_guarded', pub_qos)

        self.current_velocity = 0.0
        self.current_pose = None
        self.last_odom_time = self.get_clock().now()
        self.last_selected_curvature = 0.0
        self.latest_unchecked_traj = None 

        # 충돌 상태 플래그
        self.is_collided = 0 

        self.init_logger()

    def init_logger(self):
        file_name = f"safety_guard_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_path = os.path.join('/workspace', file_name)
        
        headers = [
            "Timestamp",
            "Ego_Velocity",           
            "Selected_Rank",          
            "Guard_Intervened",       
            "Unchecked_Planned_Vel",  
            "Selected_Planned_Vel",   
            "Unchecked_Curvature",    
            "Selected_Curvature",     
            "Is_Emergency_Stop",
            "System_Delay",
            "Is_Collided"  # [NEW] 충돌 여부 (0: 정상, 1: 쾅!)
        ]

        try:
            self.csv_file = open(self.log_path, 'w', newline='')
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(headers)
            self.get_logger().info(f"==== LOG FILE: {self.log_path} ====")
        except Exception as e:
            self.get_logger().warn(f"Log Error: {e}")

    # [NEW] 충돌 시 호출되는 함수
    def on_collision(self, msg):
        self.is_collided = 1
        self.get_logger().error(f"💥 COLLISION DETECTED! Object: {msg.other_actor_id}")

    def on_odom(self, msg):
        self.current_velocity = msg.twist.twist.linear.x
        self.current_pose = msg.pose.pose
        self.last_odom_time = self.get_clock().now()
    
    def on_unchecked(self, msg):
        self.latest_unchecked_traj = msg

    def calc_avg_vel(self, traj):
        if len(traj.points) == 0: return 0.0
        total = sum([p.longitudinal_velocity_mps for p in traj.points])
        return total / len(traj.points)

    def on_candidates(self, msg):
        selected_rank = -1
        guard_intervened = 0
        is_emergency = 0
        
        unchecked_vel = 0.0
        unchecked_curv = 0.0
        selected_planned_vel = 0.0
        final_curvature = 0.0

        if (self.get_clock().now() - self.last_odom_time).nanoseconds > 1.0 * 1e9:
            self.publish_stop(msg)
            return

        delay = self.get_parameter('system_delay').value
        a_max = self.get_parameter('max_decel').value
        lat_max = self.get_parameter('max_lat_acc').value
        
        if self.latest_unchecked_traj is not None:
            unchecked_vel = self.calc_avg_vel(self.latest_unchecked_traj)
            unchecked_curv = self.get_avg_curvature(self.latest_unchecked_traj)
        
        feasible_trajs = [] 
        sorted_trajs = msg.candidate_trajectories
        
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
            best_candidate = min(
                feasible_trajs,
                key=lambda x: abs(x['curvature'] - self.last_selected_curvature)
            )
            output.points = best_candidate['trajectory'].points
            self.last_selected_curvature = best_candidate['curvature']
            selected_rank = best_candidate['original_idx']
            
            selected_planned_vel = self.calc_avg_vel(best_candidate['trajectory'])
            final_curvature = best_candidate['curvature']
            
            if unchecked_vel > 1.0 and selected_planned_vel < 0.5:
                guard_intervened = 1
        else:
            output = self.generate_emergency_stop_path(output_header)
            self.last_selected_curvature = 0.0
            is_emergency = 1
            guard_intervened = 1
            selected_planned_vel = 0.0

        self.pub_trajectory.publish(output)
        
        # [데이터 기록]
        try:
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.writer.writerow([
                current_time,
                self.current_velocity,
                selected_rank,
                guard_intervened,
                f"{unchecked_vel:.2f}",
                f"{selected_planned_vel:.2f}",
                f"{unchecked_curv:.4f}",
                f"{final_curvature:.4f}",
                is_emergency,
                delay,
                self.is_collided # 여기에 충돌 여부 기록 (1이면 사고!)
            ])
            
            # 충돌 플래그는 한 번 기록 후 초기화 (선택 사항)
            # self.is_collided = 0 
        except Exception:
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