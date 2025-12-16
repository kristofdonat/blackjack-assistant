import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path
import csv
from datetime import datetime
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

# -----------------------------
#  POLICY MODEL: "Should we draw?" (x1 = card count, x2 = total value)
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
        return net(x).item()


# -----------------------------
#  CARD VALUE HANDLING (Blackjack rules)
# -----------------------------

RANK_VALUE_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 10, "Q": 10, "K": 10,
    "A": 11,
}


def compute_hand_value(ranks: List[str]) -> int:
    """Standard blackjack scoring with Ace adjustment."""
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
#  LABEL PARSING: label -> (rank, suit)
#  Your classes look like: '10c', 'Ah', 'Ks', ...
# -----------------------------
def label_to_card(label: str) -> Tuple[str, Optional[str]]:
    """
    Convert YOLO class name to (rank, suit).
    Examples: '10c' -> ('10','c'), 'Ah' -> ('A','h')
    """
    s = label.strip()
    if len(s) < 2:
        return s.upper(), None

    suit = s[-1].lower()  # c/d/h/s
    rank = s[:-1].upper()  # '10','A','K','Q','J','2'...
    if suit not in ("c", "d", "h", "s"):
        # If for some reason suit isn't present, keep rank only
        return s.upper(), None

    return rank, suit


def card_to_string(rank: str, suit: Optional[str]) -> str:
    """Format a card for CSV logging."""
    return f"{rank}{suit}" if suit else rank


# -----------------------------
#  HAND STATE (stable, only grows during a hand)
# -----------------------------
def init_hand_state():
    return {
        # tid -> {seen, card_votes, last_seen, last_center, confirmed_card}
        "tracks": {},
        # Tracks already used to add a card
        "hand_track_ids": set(),
        # Cards in the hand: list of (rank, suit)
        "hand_cards": [],
        # Fast membership check: set of (rank, suit)
        "hand_card_keys": set(),
        # Centers of accepted cards to suppress duplicate tracks of the same physical card
        "confirmed_centers": {},  # tid -> (cx, cy)
    }


def card_already_in_hand(hand_state, rank: str, suit: Optional[str]) -> bool:
    """
    Decide if this (rank,suit) is already present in the current hand.
    If suit is None, we conservatively treat ANY same-rank as present.
    """
    if suit is None:
        return any(r == rank for (r, _s) in hand_state["hand_card_keys"])
    return (rank, suit) in hand_state["hand_card_keys"]


def update_hand_from_tracking(
    results,
    yolo_model,
    frame_idx: int,
    frame_shape,
    hand_state: dict,
    min_conf: float = 0.5,
    stable_frames: int = 5,
    vote_ratio: float = 0.7,
    drop_frames: int = 30,
    min_center_dist_ratio: float = 0.08,
):
    """
    Update hand state using YOLO tracking output.

    Returns:
      ranks_sorted: current hand ranks (sorted)
      new_card_added: True if a NEW card was added in this frame
      visible_any: True if at least one detection >= min_conf is visible in this frame
    """
    h, w = frame_shape[:2]
    min_center_dist = max(h, w) * min_center_dist_ratio

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        ranks_sorted = sorted([r for (r, _s) in hand_state["hand_cards"]])
        return ranks_sorted, False, False

    # Must have tracking IDs for stable per-object logic
    if boxes.id is None:
        ranks_sorted = sorted([r for (r, _s) in hand_state["hand_cards"]])
        return ranks_sorted, False, True

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

        rank, suit = label_to_card(label)

        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        tr = hand_state["tracks"].setdefault(
            tid,
            {
                "seen": 0,
                "card_votes": {},      # (rank,suit) -> count
                "last_seen": frame_idx,
                "last_center": (cx, cy),
                "confirmed_card": None,  # (rank,suit)
            },
        )

        tr["seen"] += 1
        tr["last_seen"] = frame_idx
        tr["last_center"] = (cx, cy)

        card_key = (rank, suit)
        tr["card_votes"][card_key] = tr["card_votes"].get(card_key, 0) + 1

        # Confirm a stable card identity for this track
        if tr["confirmed_card"] is None and tr["seen"] >= stable_frames:
            best_card, best_cnt = max(tr["card_votes"].items(), key=lambda kv: kv[1])
            if best_cnt / tr["seen"] >= vote_ratio:
                tr["confirmed_card"] = best_card
                best_rank, best_suit = best_card

                # Suppress duplicate tracks of the same physical card using center distance
                # (This is a secondary safety net; main dedupe is by (rank,suit) uniqueness.)
                too_close = False
                for _, (ex, ey) in hand_state["confirmed_centers"].items():
                    if math.dist((cx, cy), (ex, ey)) < min_center_dist:
                        too_close = True
                        break
                if too_close:
                    continue

                # Add as a new card only if not already present in this hand (rank+suit)
                if (tid not in hand_state["hand_track_ids"]) and (not card_already_in_hand(hand_state, best_rank, best_suit)):
                    hand_state["hand_track_ids"].add(tid)
                    hand_state["hand_cards"].append((best_rank, best_suit))
                    hand_state["hand_card_keys"].add((best_rank, best_suit))
                    hand_state["confirmed_centers"][tid] = (cx, cy)
                    new_card_added = True

    # Cleanup old tracks from memory (does not remove cards from the hand)
    old = [tid for tid, tr in hand_state["tracks"].items() if frame_idx - tr["last_seen"] > drop_frames]
    for tid in old:
        hand_state["tracks"].pop(tid, None)

    ranks_sorted = sorted([r for (r, _s) in hand_state["hand_cards"]])
    return ranks_sorted, new_card_added, visible_any


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

        # Tracking inference (ByteTrack / BoT-SORT)
        results = yolo_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",  # or "botsort.yaml" if IDs are unstable
            conf=MIN_CONF,
            verbose=False
        )[0]

        ranks_sorted, new_added, visible_any = update_hand_from_tracking(
            results, yolo_model, frame_idx, frame.shape, hand_state,
            min_conf=MIN_CONF,
            stable_frames=STABLE_FRAMES,
            vote_ratio=VOTE_RATIO,
            min_center_dist_ratio=MIN_CENTER_DIST_RATIO,
        )

        # End hand / reset only after sustained "no cards"
        if not visible_any:
            no_card_frames += 1
            if no_card_frames >= RESET_FRAMES:
                if current_hand_active:
                    final_cards = list(hand_state["hand_cards"])
                    final_ranks = sorted([r for (r, _s) in final_cards])

                    card_count = len(final_ranks)
                    total_value = compute_hand_value(final_ranks)
                    prob_draw = predict_draw(policy_net, card_count, total_value) if card_count > 0 else 0.0
                    decision = "YES" if prob_draw >= 0.5 else "NO"

                    # Full identities for CSV (rank+suit), e.g. "Kc Kd"
                    cards_full_str = " ".join(sorted([card_to_string(r, s) for (r, s) in final_cards]))

                    print(f"--- End of hand {current_hand_id} ---")

                    hands_log.append({
                        "hand_id": current_hand_id,
                        "cards_ranks": " ".join(final_ranks),
                        "cards_full": cards_full_str,
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

            # Only print / compute when a NEW card is added
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

                # Display only ranks (no suit on screen)
                last_display_info = (card_count, total_value, prob_draw, decision, ranks_sorted)

        # Visualization
        annotated_frame = results.plot()

        if last_display_info is not None:
            card_count, total_value, prob_draw, decision, ranks_sorted = last_display_info
            cv2.putText(
                annotated_frame,
                f"Cards: {card_count}, Total: {total_value}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Draw?: {decision} ({prob_draw:.2f})",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Cards: {' '.join(ranks_sorted)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(f"Blackjack card detection (YOLO) | model: {model_weights}", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Write CSV (one row per hand)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hand_id",
            "cards_ranks",
            "cards_full",
            "card_count",
            "total_value",
            "draw_probability",
            "draw_decision",
        ])
        for row in hands_log:
            writer.writerow([
                row["hand_id"],
                row["cards_ranks"],
                row["cards_full"],
                row["card_count"],
                row["total_value"],
                f"{row['draw_probability']:.4f}",
                row["draw_decision"],
            ])

    print(f"Log file saved to: {log_path}")


# -----------------------------
#  CLI ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (0 = camera, 'video.mp4' = file)")
    parser.add_argument("--weights", type=str, default="training/yolo/yolov8/yolov8m_e100.pt", help="Path to YOLO model weights file")
    parser.add_argument("--policy", type=str, default="training/torch/blackjack_model.pth", help="Path to PyTorch policy model (draw / no draw)")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory where log CSV will be saved")
    args = parser.parse_args()

    # Source may be int (webcam) or string (file)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    main(source, args.weights, args.policy, args.logdir)
