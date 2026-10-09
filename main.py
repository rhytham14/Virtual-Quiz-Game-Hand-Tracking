import cv2
import mediapipe as mp
import csv
import time
import os
import winsound
import pygame

pygame.mixer.init()


# =========================================================
# LOAD QUESTIONS
# =========================================================

questions_file = "questions.csv"

with open(questions_file, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    questions = list(reader)


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()


# =========================================================
# FULL SCREEN WINDOW
# =========================================================

window_name = "Virtual Quiz Game"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# =========================================================
# GAME SETTINGS
# =========================================================

question_number = 0
score = 0

# Give the user 20 seconds per question
QUESTION_TIME = 20

# Finger must remain over option for this long
SELECTION_TIME = 1.5

question_start_time = time.monotonic()

finish_sound_played = False

feedback = ""
feedback_start_time = 0

game_over = False
finish_sound_played = False

# Used to prevent repeated countdown sounds
last_countdown = None

# Used for answer hovering
hovered_option = None
hover_start_time = None

# Smooth finger movement
smooth_x = None
smooth_y = None

SMOOTHING = 0.25


# =========================================================
# SOUND FUNCTIONS
# =========================================================

def correct_sound():
    winsound.Beep(900, 150)
    winsound.Beep(1200, 180)


def wrong_sound():
    winsound.Beep(300, 250)
    winsound.Beep(220, 300)


def countdown_sound():
    winsound.Beep(700, 100)


def timeout_sound():
    winsound.Beep(250, 500)


def finish_sound():
    audio_path = os.path.join(
        os.path.dirname(__file__),
        "yay.mp3"
    )

    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()


# =========================================================
# NEXT QUESTION
# =========================================================

def next_question():

    global question_number
    global question_start_time
    global feedback
    global hovered_option
    global hover_start_time
    global last_countdown

    question_number += 1

    question_start_time = time.monotonic()

    feedback = ""

    hovered_option = None
    hover_start_time = None

    last_countdown = None


# =========================================================
# MEDIAPIPE
# =========================================================

with HandLandmarker.create_from_options(options) as landmarker:

    frame_timestamp = 0

    while True:

        # =================================================
        # READ CAMERA
        # =================================================

        success, frame = cap.read()

        if not success:
            print("Could not read webcam")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape


        # =================================================
        # GAME COMPLETE
        # =================================================

        if question_number >= len(questions):
            if not finish_sound_played:
                finish_sound()
                finish_sound_played = True

            frame[:] = (25, 25, 25)

            cv2.putText(
                frame,
                "QUIZ COMPLETE!",
                (int(width * 0.28), int(height * 0.30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 255, 0),
                5
            )

            cv2.putText(
                frame,
                f"FINAL SCORE: {score}/{len(questions)}",
                (int(width * 0.30), int(height * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                4
            )

            cv2.putText(
                frame,
                "Press R to Restart",
                (int(width * 0.34), int(height * 0.60)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "Press Q to Quit",
                (int(width * 0.36), int(height * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):

                question_number = 0
                score = 0
                question_start_time = time.monotonic()

                hovered_option = None
                hover_start_time = None
                feedback = ""

                finish_sound()

            continue


        # =================================================
        # CURRENT QUESTION
        # =================================================

        current = questions[question_number]

        question = current["question"]

        options_list = [
            current["option1"],
            current["option2"],
            current["option3"],
            current["option4"]
        ]

        correct_answer = int(current["answer"])


        # =================================================
        # TIME
        # =================================================

        elapsed = time.monotonic() - question_start_time

        remaining = max(
            0,
            QUESTION_TIME - int(elapsed)
        )


        # =================================================
        # COUNTDOWN SOUND
        # =================================================

        if remaining <= 5 and remaining > 0:

            if last_countdown != remaining:

                countdown_sound()

                last_countdown = remaining


        # =================================================
        # MEDIAPIPE IMAGE
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )


        # =================================================
        # TOP INFORMATION AREA
        # =================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 150),
            (35, 35, 35),
            -1
        )


        # =================================================
        # QUESTION NUMBER
        # =================================================

        cv2.putText(
            frame,
            f"QUESTION {question_number + 1}/{len(questions)}",
            (40, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # =================================================
        # SCORE
        # =================================================

        cv2.putText(
            frame,
            f"SCORE: {score}",
            (width - 220, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # =================================================
        # TIMER
        # =================================================

        timer_color = (0, 255, 255)

        if remaining <= 5:
            timer_color = (0, 0, 255)

        cv2.putText(
            frame,
            f"TIME: {remaining}",
            (width - 220, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            timer_color,
            2
        )


        # =================================================
        # QUESTION TEXT
        # =================================================

        if len(question) > 65:

            split_position = question.rfind(" ", 0, 65)

            if split_position == -1:
                split_position = 65

            line1 = question[:split_position]
            line2 = question[split_position:]

            cv2.putText(
                frame,
                line1,
                (40, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                line2,
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                question,
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


        # =================================================
        # ANSWER BOX POSITIONS
        # =================================================

        margin_x = int(width * 0.08)

        box_width = int(width * 0.36)

        box_height = int(height * 0.13)

        gap_x = int(width * 0.08)

        y1 = int(height * 0.42)

        y2 = int(height * 0.62)

        answer_boxes = [

            (
                margin_x,
                y1,
                margin_x + box_width,
                y1 + box_height
            ),

            (
                margin_x + box_width + gap_x,
                y1,
                margin_x + box_width * 2 + gap_x,
                y1 + box_height
            ),

            (
                margin_x,
                y2,
                margin_x + box_width,
                y2 + box_height
            ),

            (
                margin_x + box_width + gap_x,
                y2,
                margin_x + box_width * 2 + gap_x,
                y2 + box_height
            )
        ]


        # =================================================
        # DRAW ANSWER BOXES
        # =================================================

        for i in range(4):

            x1, box_y1, x2, box_y2 = answer_boxes[i]

            thickness = 3

            # Highlight currently hovered option
            if hovered_option == i:

                cv2.rectangle(
                    frame,
                    (x1, box_y1),
                    (x2, box_y2),
                    (0, 255, 255),
                    6
                )

            else:

                cv2.rectangle(
                    frame,
                    (x1, box_y1),
                    (x2, box_y2),
                    (255, 255, 255),
                    thickness
                )

            text = f"{i + 1}. {options_list[i]}"

            cv2.putText(
                frame,
                text,
                (x1 + 20, box_y1 + int(box_height * 0.60)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        # =================================================
        # HAND DETECTION
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            index_tip = hand[8]

            # Raw finger position
            raw_x = int(index_tip.x * width)
            raw_y = int(index_tip.y * height)

# Initialize smoothing
            if smooth_x is None:

                smooth_x = raw_x
                smooth_y = raw_y

            # Smooth movement
            smooth_x = smooth_x + SMOOTHING * (raw_x - smooth_x)
            smooth_y = smooth_y + SMOOTHING * (raw_y - smooth_y)

            finger_x = int(smooth_x)
            finger_y = int(smooth_y)


            # Draw finger cursor
            cv2.circle(
                frame,
                (finger_x, finger_y),
                15,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                (finger_x, finger_y),
                20,
                (255, 255, 255),
                2
            )


            # =============================================
            # FIND HOVERED OPTION
            # =============================================

            current_hover = None

            for i in range(4):

                x1, box_y1, x2, box_y2 = answer_boxes[i]

                if (
                    x1 < finger_x < x2
                    and
                    box_y1 < finger_y < box_y2
                ):

                    current_hover = i
                    break


            # =============================================
            # START / CONTINUE HOVER TIMER
            # =============================================

            if current_hover != hovered_option:

                hovered_option = current_hover

                if current_hover is not None:

                    hover_start_time = time.monotonic()

                else:

                    hover_start_time = None


            # =============================================
            # SHOW SELECTION PROGRESS
            # =============================================

            if (
                hovered_option is not None
                and
                hover_start_time is not None
                and
                not feedback
            ):

                hover_elapsed = (
                    time.monotonic()
                    - hover_start_time
                )

                progress = min(
                    hover_elapsed / SELECTION_TIME,
                    1.0
                )

                # Progress bar
                x1, box_y1, x2, box_y2 = answer_boxes[
                    hovered_option
                ]

                progress_width = int(
                    (x2 - x1) * progress
                )

                cv2.rectangle(
                    frame,
                    (x1, box_y2 + 8),
                    (x1 + progress_width, box_y2 + 15),
                    (0, 255, 255),
                    -1
                )

                # Select after 1.5 seconds
                if hover_elapsed >= SELECTION_TIME:

                    selected_answer = hovered_option + 1

                    if selected_answer == correct_answer:

                        score += 1
                        feedback = "CORRECT!"

                        correct_sound()

                    else:

                        feedback = "WRONG!"

                        wrong_sound()

                    feedback_start_time = time.monotonic()


        # =================================================
        # FEEDBACK
        # =================================================

        if feedback:

            if feedback == "CORRECT!":

                cv2.putText(
                    frame,
                    "CORRECT!",
                    (
                        int(width * 0.40),
                        int(height * 0.88)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    4
                )

            elif feedback == "WRONG!":

                cv2.putText(
                    frame,
                    "WRONG!",
                    (
                        int(width * 0.43),
                        int(height * 0.88)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    4
                )

            elif feedback == "TIME UP!":

                cv2.putText(
                    frame,
                    "TIME UP!",
                    (
                        int(width * 0.40),
                        int(height * 0.88)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 165, 255),
                    4
                )


            # Wait before next question
            if time.monotonic() - feedback_start_time > 1.5:

                next_question()


        # =================================================
        # TIME OUT
        # =================================================

        if (
            elapsed >= QUESTION_TIME
            and
            not feedback
        ):

            feedback = "TIME UP!"

            feedback_start_time = time.monotonic()

            timeout_sound()


        # =================================================
        # INSTRUCTIONS
        # =================================================

        cv2.putText(
            frame,
            "Move your index finger over an answer and hold",
            (
                int(width * 0.22),
                int(height * 0.96)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )


        # =================================================
        # SHOW GAME
        # =================================================

        cv2.imshow(
            window_name,
            frame
        )


        # =================================================
        # KEYBOARD CONTROLS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()