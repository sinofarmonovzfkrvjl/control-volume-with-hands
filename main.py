import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

# Initialize Mediapipe Hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Start capturing video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the coordinates of the index finger tip, thumb tip, and middle finger tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                
                # Convert to pixel coordinates
                h, w, c = frame.shape
                ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                mx, my = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                mcp_x, mcp_y = int(middle_finger_mcp.x * w), int(middle_finger_mcp.y * h)
                rx, ry = int(ring_finger_tip.x * w), int(ring_finger_tip.y * h)
                px, py = int(pinky_finger_tip.x * w), int(pinky_finger_tip.y * h)
                
                # Check if the middle finger is raised
                if my < mcp_y:
                    # Calculate the distance between index finger tip and thumb tip
                    distance = np.sqrt((ix - tx)**2 + (iy - ty)**2)
                    
                    # Normalize the distance to a volume level
                    vol = np.interp(distance, [50, 200], [min_vol, max_vol])
                    
                    # Set the system volume
                    volume.SetMasterVolumeLevel(vol, None)
                    
                    # Display the volume level on the frame
                    cv2.putText(frame, f'Volume: {int((vol - min_vol) / (max_vol - min_vol) * 100)}%', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw circles on the index finger tip, thumb tip, and middle finger tip
                cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (tx, ty), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (mx, my), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (rx, ry), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (px, py), 10, (0, 255, 0), cv2.FILLED)
        
        # Display the resulting frame
        cv2.imshow('Hand Gesture Volume Control', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
