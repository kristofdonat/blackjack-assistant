import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path
import csv
from datetime import datetime

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
            nn.Sigmoid()  # binary output: probability of drawing
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
    else:
        # first letter for faces/ace (J,Q,K,A,...)
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
#  MAIN LOGIC
# -----------------------------

def main(video_source, model_weights, policy_model_path, log_dir):
    # Load YOLO model
    yolo_model = YOLO(model_weights)
    yolo_model.conf = 0.5  # confidence threshold (tune if needed)

    # Load decision (draw?) model
    policy_net = load_policy_model(policy_model_path)

    # Prepare logging
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"blackjack_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Structure: list of dict states
    all_states = []  # each row corresponds to one "snapshot" during a hand

    # Hand tracking
    current_hand_id = 0
    current_hand_active = False
    last_cards_snapshot = []
    no_card_frames = 0
    NO_CARD_FRAMES_TO_END_HAND = 10  # adjust (depends on FPS)

    # Last display information
    last_display_info = None  # (card_count, total_value, prob, decision_str, cards)

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

        # YOLO inference (single result)
        results = yolo_model(frame, verbose=False)[0]

        # Extract labels from detections
        detected_labels = []
        if results.boxes is not None and len(results.boxes) > 0:
            for cls_idx in results.boxes.cls:
                cls_idx = int(cls_idx.item())
                label = yolo_model.names[cls_idx]
                detected_labels.append(label)

        # Convert labels -> ranks
        ranks = [label_to_rank(lbl) for lbl in detected_labels]

        # Determine if we see any cards
        if len(ranks) == 0:
            # No cards in this frame
            if current_hand_active:
                no_card_frames += 1
                if no_card_frames >= NO_CARD_FRAMES_TO_END_HAND:
                    # End current hand
                    print(f"--- End of hand {current_hand_id} ---")
                    current_hand_active = False
                    last_cards_snapshot = []
                    last_display_info = None
            else:
                # No hand active; nothing to do
                pass
        else:
            # We see at least one card
            no_card_frames = 0
            ranks_sorted = sorted(ranks)

            # Start a new hand if there is none active
            if not current_hand_active:
                current_hand_id += 1
                current_hand_active = True
                print(f"=== New hand started: {current_hand_id} ===")

            # If cards changed compared to last snapshot, create new state
            if ranks_sorted != last_cards_snapshot:
                last_cards_snapshot = ranks_sorted

                card_count = len(ranks_sorted)
                total_value = compute_hand_value(ranks_sorted)
                prob_draw = predict_draw(policy_net, card_count, total_value)
                decision = "YES" if prob_draw >= 0.5 else "NO"

                # Log to console
                print(
                    f"Hand {current_hand_id} | cards: {ranks_sorted} | "
                    f"count: {card_count} | total: {total_value} | "
                    f"draw?: {decision} ({prob_draw:.2f})"
                )

                # Save state (for final CSV)
                state_row = {
                    "hand_id": current_hand_id,
                    "cards": " ".join(ranks_sorted),
                    "card_count": card_count,
                    "total_value": total_value,
                    "draw_probability": prob_draw,
                    "draw_decision": decision,
                }
                all_states.append(state_row)

                # Update display info
                last_display_info = (
                    card_count,
                    total_value,
                    prob_draw,
                    decision,
                    ranks_sorted,
                )

        # Annotate frame with YOLO + text
        annotated_frame = results.plot()  # YOLO draws boxes and labels

        if last_display_info is not None:
            card_count, total_value, prob_draw, decision, ranks_sorted = last_display_info
            text1 = f"Cards: {card_count}, Total: {total_value}"
            text2 = f"Draw?: {decision} ({prob_draw:.2f})"
            text3 = f"Cards: {' '.join(ranks_sorted)}"

            cv2.putText(
                annotated_frame,
                text1,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                text2,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                text3,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(
            f"Blackjack card detection (YOLOv8) | model: {model_weights}",
            annotated_frame,
        )

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Finalize: write CSV log
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "hand_id",
                "cards",
                "card_count",
                "total_value",
                "draw_probability",
                "draw_decision",
            ]
        )
        for row in all_states:
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
