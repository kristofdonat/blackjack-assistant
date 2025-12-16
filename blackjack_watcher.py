import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path
import csv
from datetime import datetime
import math

import torch
import torch.nn as nn

# -----------------------------
#  ML MODEL: "Húzzunk-e?" (x1 = card count, x2 = total value)
# -----------------------------
class BlackjackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def load_policy_model(path: str) -> BlackjackNet:
    net = BlackjackNet()
    state = torch.load(path, weights_only=True, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net


def predict_draw(net: BlackjackNet, card_count: int, total_value: int) -> float:
    """
    Returns probability that we should draw another card (0..1).
    x1 = card_count, x2 = total_value
    """
    with torch.no_grad():
        x = torch.tensor([[float(card_count), float(total_value)]])
        prob = net(x).item()
    return prob


# -----------------------------
#  CARD VALUE HANDLING (Blackjack rules)
# -----------------------------

RANK_VALUE_MAP = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,  # initially 11, we will adjust for bust
}


def label_to_rank(label: str) -> str:
    """
    Converts a YOLO class name to a rank.
    Assumptions (adapt if needed):
      - Labels like: "2", "3", ..., "10", "J", "Q", "K", "A"
      - Or labels like: "10H", "AS", "KD" -> the rank is the numeric / first letter(s).
    Adjust this depending on how you named your YOLO classes.
    """
    label = label.strip().upper()
    # numeric at the start (2..10)
    if label[0].isdigit():
        digits = ""
        for ch in label:
            if ch.isdigit():
                digits += ch
            else:
                break
        return digits  # e.g. "10"
    return label[0]  # "A", "K", etc.


def compute_hand_value(ranks):
    """
    Standard blackjack value:
    - Aces start as 11
    - While total > 21 and we have aces counted as 11, subtract 10 for each.
    """
    total = 0
    ace_count = 0

    for r in ranks:
        v = RANK_VALUE_MAP.get(r, 0)
        total += v
        if r == "A":
            ace_count += 1

    while total > 21 and ace_count > 0:
        total -= 10
        ace_count -= 1

    return total


# -----------------------------
#  HAND STATE (stabil, only grows)
# -----------------------------
def init_hand_state():
    return {
        "tracks": {},             # tid -> {seen, rank_votes, last_seen, last_center, confirmed_rank}
        "hand_track_ids": set(),  # what tids are already in hand
        "hand_ranks": [],         #hand (only grows)
        "hand_rank_counts": {},   # rank -> count in hand (max_per_rank enforcement)
        "confirmed_centers": {},  # tid -> (cx, cy) for sorting duplicates
    }


def update_hand_from_tracking(
    results,
    yolo_model,
    frame_idx: int,
    frame_shape,
    hand_state: dict,
    min_conf: float = 0.5,
    stable_frames: int = 5,
    vote_ratio: float = 0.7,
    max_per_rank: int = 1,
    drop_frames: int = 30,
    min_center_dist_ratio: float = 0.08,
):
    """
    Return:
      ranks_sorted: a jelenlegi hand (csak bővül), rendezve
      new_card_added: True ha MOST került be új kártya a hand-be
      visible_any: True ha ebben a frame-ben láttunk legalább 1 (min_conf feletti) detekciót
    """
    h, w = frame_shape[:2]
    min_center_dist = max(h, w) * min_center_dist_ratio

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return sorted(hand_state["hand_ranks"]), False, False

    # tracking id-s
    if boxes.id is None:
        # without tracking, we cannot maintain hand state
        return sorted(hand_state["hand_ranks"]), False, True

    visible_any = False
    new_card_added = False

    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf < min_conf:
            continue

        visible_any = True

        tid = int(boxes.id[i].item())
        cls_id = int(boxes.cls[i].item())
        label = yolo_model.names[cls_id]
        rank = label_to_rank(label)

        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        tr = hand_state["tracks"].setdefault(
            tid,
            {
                "seen": 0,
                "rank_votes": {},
                "last_seen": frame_idx,
                "last_center": (cx, cy),
                "confirmed_rank": None,
            },
        )
        tr["seen"] += 1
        tr["last_seen"] = frame_idx
        tr["last_center"] = (cx, cy)
        tr["rank_votes"][rank] = tr["rank_votes"].get(rank, 0) + 1

        # stable enough to confirm rank?
        if tr["confirmed_rank"] is None and tr["seen"] >= stable_frames:
            best_rank, best_cnt = max(tr["rank_votes"].items(), key=lambda kv: kv[1])
            if best_cnt / tr["seen"] >= vote_ratio:
                tr["confirmed_rank"] = best_rank

                # sort duplicated tracks if they are too close to each other
                too_close = False
                for _, (ex, ey) in hand_state["confirmed_centers"].items():
                    if math.dist((cx, cy), (ex, ey)) < min_center_dist:
                        too_close = True
                        break

                # only add to hand once
                current_cnt = hand_state["hand_rank_counts"].get(best_rank, 0)
                if (not too_close) and (tid not in hand_state["hand_track_ids"]) and (current_cnt < max_per_rank):
                    hand_state["hand_track_ids"].add(tid)
                    hand_state["hand_ranks"].append(best_rank)
                    hand_state["hand_rank_counts"][best_rank] = current_cnt + 1
                    hand_state["confirmed_centers"][tid] = (cx, cy)
                    new_card_added = True

    # clean old tracks
    old = [tid for tid, tr in hand_state["tracks"].items() if frame_idx - tr["last_seen"] > drop_frames]
    for tid in old:
        hand_state["tracks"].pop(tid, None)

    return sorted(hand_state["hand_ranks"]), new_card_added, visible_any


# -----------------------------
#  MAIN LOGIC
# -----------------------------

def main(video_source, model_weights, policy_model_path, log_dir):
    # Load YOLO model
    yolo_model = YOLO(model_weights)

    policy_net = load_policy_model(policy_model_path)

    # Prepare logging
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"blackjack_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # log data per hand
    hands_log = []

    # Hand tracking
    current_hand_id = 0
    current_hand_active = False
    no_card_frames = 0

    # tuning
    RESET_FRAMES = 10
    MIN_CONF = 0.5
    STABLE_FRAMES = 5
    VOTE_RATIO = 0.7
    MIN_CENTER_DIST_RATIO = 0.08

    # Last display information
    last_display_info = None  # (card_count, total_value, prob, decision, ranks_sorted)
    hand_state = init_hand_state()
    frame_idx = 0

    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Unable to open video source: {video_source}")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # YOLO tracking (BoT-SORT default, ByteTrack if tracker=bytetrack.yaml) :contentReference[oaicite:1]{index=1}
        results = yolo_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",  # or "botsort.yaml"
            conf=MIN_CONF,
            verbose=False
        )[0]

        ranks_sorted, new_added, visible_any = update_hand_from_tracking(
            results, yolo_model, frame_idx, frame.shape, hand_state,
            min_conf=MIN_CONF,
            stable_frames=STABLE_FRAMES,
            vote_ratio=VOTE_RATIO,
            min_center_dist_ratio=MIN_CENTER_DIST_RATIO,
            max_per_rank=1,
        )

        # end of hand detection
        if not visible_any:
            no_card_frames += 1
            if no_card_frames >= RESET_FRAMES:
                if current_hand_active:
                    # close out current hand
                    final_cards = sorted(hand_state["hand_ranks"])
                    card_count = len(final_cards)
                    total_value = compute_hand_value(final_cards)
                    prob_draw = predict_draw(policy_net, card_count, total_value) if card_count > 0 else 0.0
                    decision = "YES" if prob_draw >= 0.5 else "NO"

                    print(f"--- End of hand {current_hand_id} ---")

                    hands_log.append({
                        "hand_id": current_hand_id,
                        "cards": " ".join(final_cards),
                        "card_count": card_count,
                        "total_value": total_value,
                        "draw_probability": prob_draw,
                        "draw_decision": decision,
                    })

                current_hand_active = False
                last_display_info = None
                hand_state = init_hand_state()
        else:
            no_card_frames = 0

            if not current_hand_active:
                current_hand_id += 1
                current_hand_active = True
                print(f"=== New hand started: {current_hand_id} ===")

            # only calculate and display when a new card is added
            if new_added:
                card_count = len(ranks_sorted)
                total_value = compute_hand_value(ranks_sorted)
                prob_draw = predict_draw(policy_net, card_count, total_value)
                decision = "YES" if prob_draw >= 0.5 else "NO"

                print(
                    f"Hand {current_hand_id} | cards: {ranks_sorted} | "
                    f"count: {card_count} | total: {total_value} | "
                    f"draw?: {decision} ({prob_draw:.2f})"
                )

                last_display_info = (card_count, total_value, prob_draw, decision, ranks_sorted)

        # Annotate frame with YOLO + text
        annotated_frame = results.plot()  # YOLO draws boxes and labels

        if last_display_info is not None:
            card_count, total_value, prob_draw, decision, ranks_sorted = last_display_info
            cv2.putText(annotated_frame, f"Cards: {card_count}, Total: {total_value}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Draw?: {decision} ({prob_draw:.2f})",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Cards: {' '.join(ranks_sorted)}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(f"Blackjack card detection (YOLO) | model: {model_weights}", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Finalize: write CSV log
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hand_id", "cards", "card_count", "total_value", "draw_probability", "draw_decision"])
        for row in hands_log:
            writer.writerow(
                [
                row["hand_id"],
                row["cards"],
                row["card_count"],
                row["total_value"],
                f"{row['draw_probability']:.4f}",
                row["draw_decision"],
                ]
            )

    print(f"Log file saved to: {log_path}")


# -----------------------------
#  CLI ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (0 = camera, 'video.mp4' = file)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8_30.pt",
        help="Path to YOLOv8 model weights file",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="training/torch/blackjack_model.pth",
        help="Path to PyTorch policy model (draw / no draw)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Directory where log CSV will be saved",
    )

    args = parser.parse_args()

    # Source may be int (webcam) or string (file)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    main(source, args.weights, args.policy, args.logdir)
