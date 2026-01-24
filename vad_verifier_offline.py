#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict, Counter
import bisect

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py


TOPIC_CANDS = "/planning/vad/candidate_trajectories"
TOPIC_BASE  = "/planning/trajectory"
TOPIC_OBJS  = "/perception/object_recognition/objects"


def stamp_to_int(stamp_msg) -> int:
    """builtin_interfaces/Time -> int nanoseconds"""
    return int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)


def hypot2(x, y):
    return math.hypot(x, y)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pick_top1_predicted_path(predicted_paths):
    """Pick predicted path with max confidence; return None if empty."""
    if not predicted_paths:
        return None
    best = max(predicted_paths, key=lambda p: float(p.confidence))
    if not best.path:
        return None
    return best


def yaw_from_quat(q):
    """Assuming flat ground: yaw from quaternion."""
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

def uuid_to_str(uuid_msg) -> str:
    # autoware UUID msg variants: sometimes 'uuid' field (uint8[16]), sometimes direct.
    if hasattr(uuid_msg, "uuid"):
        arr = list(uuid_msg.uuid)
    else:
        arr = list(uuid_msg)
    return "".join([f"{b:02x}" for b in arr])


def build_generator_map(cand_msg):
    """
    CandidateTrajectories.generator_info: list of GeneratorInfo
    each has generator_id (UUID) and generator_name (string)
    """
    gen_map = {}
    if hasattr(cand_msg, "generator_info"):
        for gi in cand_msg.generator_info:
            try:
                key = uuid_to_str(gi.generator_id)
                gen_map[key] = str(gi.generator_name)
            except Exception:
                pass
    return gen_map


def cand_name(cand, gen_map, fallback_idx=None):
    """
    CandidateTrajectory has generator_id (UUID). Map it to name.
    """
    if hasattr(cand, "generator_id"):
        key = uuid_to_str(cand.generator_id)
        if key in gen_map:
            return gen_map[key]
        return key[:8]  # short uuid fallback
    return f"idx_{fallback_idx}" if fallback_idx is not None else "unknown"



def make_obj_future_fn(obj_msg):
    """
    Build object future position function pos(t)->(x,y) for t in seconds
    using top-1 predicted_path (0.1s) then CV extension.
    """
    kin = obj_msg.kinematics
    top1 = pick_top1_predicted_path(kin.predicted_paths)

    # object size -> radius approximation
    dim_x = float(obj_msg.shape.dimensions.x)
    dim_y = float(obj_msg.shape.dimensions.y)
    r_obj = 0.5 * hypot2(dim_x, dim_y)  # conservative-ish circle

    # speed magnitude (avoid frame confusion with vx,vy)
    vx = float(kin.initial_twist_with_covariance.twist.linear.x)
    vy = float(kin.initial_twist_with_covariance.twist.linear.y)
    v_mag = hypot2(vx, vy)

    if top1 is None:
        # fallback: constant position at initial pose
        p0 = kin.initial_pose_with_covariance.pose.position
        x0, y0 = float(p0.x), float(p0.y)

        def pos(t):
            return x0, y0

        return pos, r_obj

    # predicted path details
    dt_pred = float(top1.time_step.nanosec) * 1e-9 + float(top1.time_step.sec)
    if dt_pred <= 0.0:
        dt_pred = 0.1

    path = top1.path
    n = len(path)
    t_max = (n - 1) * dt_pred

    # get last heading from predicted path to define CV direction
    if n >= 2:
        p_a = path[-2].position
        p_b = path[-1].position
        dx = float(p_b.x) - float(p_a.x)
        dy = float(p_b.y) - float(p_a.y)
        yaw = math.atan2(dy, dx) if (abs(dx) + abs(dy)) > 1e-9 else yaw_from_quat(path[-1].orientation)
    else:
        yaw = yaw_from_quat(path[-1].orientation)

    vx_map = v_mag * math.cos(yaw)
    vy_map = v_mag * math.sin(yaw)

    def pos(t):
        # Use nearest index for speed; you can interpolate if you want.
        if t <= t_max + 1e-9:
            idx = int(round(t / dt_pred))
            idx = int(clamp(idx, 0, n - 1))
            pp = path[idx].position
            return float(pp.x), float(pp.y)
        else:
            # CV rollout from last predicted point
            pp_last = path[-1].position
            xL, yL = float(pp_last.x), float(pp_last.y)
            dt = t - t_max
            return xL + vx_map * dt, yL + vy_map * dt

    return pos, r_obj


def get_candidate_xy_at_times(cand_traj, times_s):
    """
    CandidateTrajectory.points has time_from_start and pose.position.
    Here VAD gives exactly 0.0,0.5,...3.0.
    We'll map time->(x,y) by nearest time_from_start.
    """
    # build dict from time_from_start(sec) to xy
    pts = cand_traj.points
    if not pts:
        return {t: (math.nan, math.nan) for t in times_s}

    # list of (t, x, y)
    seq = []
    for p in pts:
        t = float(p.time_from_start.sec) + float(p.time_from_start.nanosec) * 1e-9
        x = float(p.pose.position.x)
        y = float(p.pose.position.y)
        seq.append((t, x, y))
    seq.sort(key=lambda a: a[0])

    out = {}
    for tq in times_s:
        # nearest neighbor
        best = min(seq, key=lambda a: abs(a[0] - tq))
        out[tq] = (best[1], best[2])
    return out


def match_baseline_to_candidate_id(baseline_traj, cand_list, n_points=4):
    """
    Determine which candidate is closest to baseline by MSE on first n_points.
    """
    bpts = baseline_traj.points[:n_points]
    if not bpts or not cand_list:
        return None

    bxys = [(float(p.pose.position.x), float(p.pose.position.y)) for p in bpts]

    best_k = None
    best_mse = float("inf")
    for k, cand in enumerate(cand_list):
        cpts = cand.points[:n_points]
        if len(cpts) < len(bxys):
            continue
        mse = 0.0
        for i, (bx, by) in enumerate(bxys):
            cx = float(cpts[i].pose.position.x)
            cy = float(cpts[i].pose.position.y)
            dx = bx - cx
            dy = by - cy
            mse += dx * dx + dy * dy
        mse /= float(len(bxys))
        if mse < best_mse:
            best_mse = mse
            best_k = k

    return best_k


def compute_metrics_for_candidates(cand_list, objects_msg, ego_radius, times_s):
    """
    For each candidate compute:
      - min_clearance over objects and times
      - min_ttc surrogate from nearest-object distance curve
    """
    # build object future fns
    obj_fns = []
    for obj in objects_msg.objects:
        fn, r_obj = make_obj_future_fn(obj)
        obj_fns.append((fn, r_obj))

    results = []
    for cand in cand_list:
        ego_xy = get_candidate_xy_at_times(cand, times_s)

        # dmin(t): distance to nearest object center at time t
        dmins = []
        clear_mins = []

        for t in times_s:
            ex, ey = ego_xy[t]
            if math.isnan(ex) or math.isnan(ey) or not obj_fns:
                dmins.append(float("inf"))
                clear_mins.append(float("inf"))
                continue

            best_d = float("inf")
            best_clear = float("inf")
            for (fn, r_obj) in obj_fns:
                ox, oy = fn(t)
                d = hypot2(ex - ox, ey - oy)
                clear = d - (ego_radius + r_obj)
                if d < best_d:
                    best_d = d
                if clear < best_clear:
                    best_clear = clear

            dmins.append(best_d)
            clear_mins.append(best_clear)

        min_clear = min(clear_mins) if clear_mins else float("inf")

        # TTC surrogate
        dt = times_s[1] - times_s[0] if len(times_s) >= 2 else 0.5
        ttc_min = float("inf")
        for i in range(len(times_s) - 1):
            d0 = dmins[i]
            d1 = dmins[i + 1]
            if not math.isfinite(d0) or not math.isfinite(d1):
                continue
            d_dot = (d1 - d0) / dt
            if d_dot < -1e-6:
                ttc = d0 / (-d_dot)
                if ttc < ttc_min:
                    ttc_min = ttc

        results.append({
            "min_clearance": float(min_clear),
            "min_ttc": float(ttc_min) if math.isfinite(ttc_min) else float("inf"),
        })

    return results


def choose_guard_candidate(metrics, clearance_thresh, ttc_thresh):
    """
    Fail-fast then re-rank.
    Returns guard_choice_id and fail_mask list.
    """
    fail = []
    for m in metrics:
        is_fail = (m["min_clearance"] < clearance_thresh) or (m["min_ttc"] < ttc_thresh)
        fail.append(is_fail)

    # score: bigger is better
    best_k = None
    best_score = -float("inf")
    for k, m in enumerate(metrics):
        if fail[k]:
            continue
        # Simple score
        score = m["min_clearance"] + 0.5 * m["min_ttc"]
        if score > best_score:
            best_score = score
            best_k = k

    # if all fail, fallback: pick max(min_ttc) then max(min_clearance)
    if best_k is None and metrics:
        best_k = max(range(len(metrics)),
                     key=lambda k: (metrics[k]["min_ttc"], metrics[k]["min_clearance"]))
    return best_k, fail


def read_bag_messages(bag_path):
    """
    Yields (topic, msg, msg_type_str, t_rosbag_ns).
    """
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in type_map:
            continue
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        yield topic, msg, type_map[topic], t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="rosbag2 folder path (contains metadata.yaml)")
    parser.add_argument("--out_csv", default="vad_metrics.csv")
    parser.add_argument("--ego_radius", type=float, default=1.2)
    parser.add_argument("--clearance_thresh", type=float, default=0.3)
    parser.add_argument("--ttc_thresh", type=float, default=0.8)
    args = parser.parse_args()

    rclpy.init(args=None)

    # Collect by header stamp for alignment
    cands_by_stamp = {}
    base_by_stamp = {}
    objs_by_stamp  = {}

    for topic, msg, _, _t in read_bag_messages(args.bag):
        if topic == TOPIC_CANDS:
            # CandidateTrajectories: has header inside each CandidateTrajectory, but wrapper msg has header too
            st = _t
            cands_by_stamp[st] = msg
        elif topic == TOPIC_BASE:
            st = _t
            base_by_stamp[st] = msg
        elif topic == TOPIC_OBJS:
            st = _t
            objs_by_stamp[st] = msg

    # Intersection of stamps
        # Align by nearest timestamps (bag time _t can differ per topic)
    cand_ts = sorted(cands_by_stamp.keys())
    base_ts = sorted(base_by_stamp.keys())
    obj_ts  = sorted(objs_by_stamp.keys())

    if not cand_ts or not base_ts or not obj_ts:
        print("Missing topics in bag (one of cand/base/objs is empty).")
        return

    def nearest_ts(sorted_list, t):
        i = bisect.bisect_left(sorted_list, t)
        if i == 0:
            return sorted_list[0]
        if i == len(sorted_list):
            return sorted_list[-1]
        before = sorted_list[i-1]
        after  = sorted_list[i]
        return after if (after - t) < (t - before) else before

    tol_ns = int(0.05 * 1e9)  # 50ms tolerance (adjust if needed)

    aligned = []
    for t in cand_ts:
        tb = nearest_ts(base_ts, t)
        to = nearest_ts(obj_ts,  t)
        if abs(tb - t) <= tol_ns and abs(to - t) <= tol_ns:
            aligned.append((t, tb, to))

    if not aligned:
        print("No aligned stamps found (nearest matching failed). Try increasing tol_ns.")
        return


    times_s = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    baseline_hist = Counter()
    guard_hist = Counter()
    baseline_hist_obj = Counter()
    guard_hist_obj = Counter()

    frames_no_obj = 0
    frames_with_obj = 0

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "stamp_ns",
            "bag_time_ns",
            "objects_count",
            "has_objects",
            "baseline_choice_idx",
            "baseline_choice_name",
            "guard_choice_idx",
            "guard_choice_name",
            "cand_idx",
            "cand_name",
            "min_clearance",
            "min_ttc",
            "failed_clear_or_ttc"
        ])

        for t_c, t_b, t_o in aligned:
            cand_msg = cands_by_stamp[t_c]
            base_msg = base_by_stamp[t_b]
            obj_msg  = objs_by_stamp[t_o]
            st = t_c
            #bag_time_ns = int(_t) if isinstance(st, int) else int(st) 

            bag_time_ns = int(st)

            obj_count = len(obj_msg.objects)
            has_objects = (obj_count > 0)

            if has_objects:
                frames_with_obj += 1
            else:
                frames_no_obj += 1

            # generator map + candidate list + candidate names
            gen_map = build_generator_map(cand_msg)
            cand_list = list(cand_msg.candidate_trajectories)
            cand_names = [cand_name(c, gen_map, fallback_idx=i) for i, c in enumerate(cand_list)]

            baseline_choice_idx = match_baseline_to_candidate_id(base_msg, cand_list, n_points=4)
            baseline_choice_name = "unknown"
            if baseline_choice_idx is not None and 0 <= baseline_choice_idx < len(cand_names):
                baseline_choice_name = cand_names[baseline_choice_idx]
                baseline_hist[baseline_choice_name] += 1
                if has_objects:
                    baseline_hist_obj[baseline_choice_name] += 1


            if len(obj_msg.objects) == 0:
                guard_choice_idx = baseline_choice_idx if baseline_choice_idx is not None else 0
                guard_choice_name = baseline_choice_name
                guard_hist[guard_choice_name] += 1
                if has_objects:
                    guard_hist_obj[guard_choice_name] += 1


                # CSV 포맷 유지용 placeholder
                metrics = [{"min_clearance": float("inf"), "min_ttc": float("inf")} for _ in cand_list]
                fail_mask = [False for _ in cand_list]

            else:
                metrics = compute_metrics_for_candidates(
                    cand_list=cand_list,
                    objects_msg=obj_msg,
                    ego_radius=args.ego_radius,
                    times_s=times_s
                )
                guard_choice_idx, fail_mask = choose_guard_candidate(
                    metrics,
                    clearance_thresh=args.clearance_thresh,
                    ttc_thresh=args.ttc_thresh
                )

                guard_choice_name = "unknown"
                if guard_choice_idx is not None and 0 <= guard_choice_idx < len(cand_names):
                    guard_choice_name = cand_names[guard_choice_idx]
                    guard_hist[guard_choice_name] += 1
                    if has_objects:
                        guard_hist_obj[guard_choice_name] += 1

            for k, m in enumerate(metrics):
                w.writerow([
                    int(st),                  # stamp_ns 
                    int(bag_time_ns),         
                    obj_count,
                    int(has_objects),
                    baseline_choice_idx if baseline_choice_idx is not None else -1,
                    baseline_choice_name,
                    guard_choice_idx if guard_choice_idx is not None else -1,
                    guard_choice_name,
                    k,
                    cand_names[k] if k < len(cand_names) else f"idx_{k}",
                    f"{m['min_clearance']:.6f}" if math.isfinite(m["min_clearance"]) else "inf",
                    f"{m['min_ttc']:.6f}" if math.isfinite(m["min_ttc"]) else "inf",
                    int(bool(fail_mask[k]))
                ])

    # Print summary
    total = len(aligned)

    print(f"Frames analyzed (all): {total}")
    print(f"Frames with objects: {frames_with_obj}")
    print(f"Frames without objects: {frames_no_obj}")

    print("\nBaseline choice histogram (ALL frames, by generator_name):")
    for name, cnt in baseline_hist.most_common():
        print(f"  {name}: {cnt} ({cnt/total*100.0:.1f}%)")

    print("\nGuard choice histogram (ALL frames, by generator_name):")
    for name, cnt in guard_hist.most_common():
        print(f"  {name}: {cnt} ({cnt/total*100.0:.1f}%)")

    if frames_with_obj > 0:
        print("\nBaseline choice histogram (ONLY frames with objects):")
        for name, cnt in baseline_hist_obj.most_common():
            print(f"  {name}: {cnt} ({cnt/frames_with_obj*100.0:.1f}%)")

        print("\nGuard choice histogram (ONLY frames with objects):")
        for name, cnt in guard_hist_obj.most_common():
            print(f"  {name}: {cnt} ({cnt/frames_with_obj*100.0:.1f}%)")

    print(f"\nSaved CSV: {args.out_csv}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
