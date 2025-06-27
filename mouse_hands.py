import cv2 #webcam
import mediapipe as mp #hand tracking
import pyautogui as pag #mouse control
import os #cleaner terminal
import time #for cooldowns

class HandTrackingController:
    def __init__(self):
        pag.FAILSAFE = True 
        pag.PAUSE = 0.01 #delay after each action

        #hand set up
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        #screen setup variables
        self.screen_width, self.screen_height = pag.size()
        self.smoothing_factor = 0.3
        self.prev_x, self.prev_y = self.screen_width // 2, self.screen_height // 2

        #action delays and thresholds
        self.pinch_threshold = 0.05
        self.was_pinching = False
        self.was_hot_toggling = False
        self.was_copying = False
        self.was_pasting = False
        self.was_backing = False

        self.last_action_time = 0
        self.last_hot_toggle_time = 0
        self.action_cooldown = 0.2
        self.hot_toggle_cooldown = 1.0

        self.prev_scroll_y = 0
        self.scroll_sensitivity = 0.5
    
        self.hot_mode = False

        self.setup_webcam()
    
    def setup_webcam(self):
        #boring webacam setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Tracking", 640, 480)
        os.system('clear')
    
    @staticmethod #static methods r useless i just felt like it
    def calculate_distance(p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    
    @staticmethod
    def is_tucked(p1, p2):
        return p1.y > p2.y
    
    def process_frame(self, frame):
        #process frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #reduces lag
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        hand_detected = False
        
        #process hands or not
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                self.process_hand_landmarks(frame, hand_landmarks)
        
        if not hand_detected:
            self.handle_no_hand_detected(frame)
        
        return frame #(the picture)
    
    def process_hand_landmarks(self, frame, hand_landmarks):
        #buit in method for drawing hands
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        #takes points from other function
        landmarks = self.extract_landmarks(hand_landmarks)

        #for smooth movement
        smoothed_x, smoothed_y = self.calculate_cursor_position(landmarks['index_tip'])

        #Gesture detection from other function
        gestures = self.detect_gestures(landmarks)

        self.update_status_display(frame, gestures)

        #hotkey handling if hotmode is activated
        if self.hot_mode:
            self.handle_hotkey_mode(gestures)
        else:
            self.handle_mouse_mode(smoothed_x, smoothed_y, gestures)
        
    def extract_landmarks(self, hand_landmarks):
        #put this in its own method so i wouldnt have to look at it
        return {
            'thumb_tip': hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            'thumb_mcp': hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP],
            'index_tip': hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            'index_pip': hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            'middle_mcp': hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            'middle_tip': hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            'ring_tip': hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            'ring_pip': hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP],
            'pinky_tip': hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP],
            'pinky_pip': hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP],
            'wrist': hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        }
    
    def calculate_cursor_position(self, index_tip):
        #smooth cursor movement to follow index tip
        x = int(index_tip.x * self.screen_width)
        y = int(index_tip.y * self.screen_height)
        
        smoothed_x = int(self.prev_x + self.smoothing_factor * (x - self.prev_x))
        smoothed_y = int(self.prev_y + self.smoothing_factor * (y - self.prev_y))
        
        self.prev_x, self.prev_y = smoothed_x, smoothed_y

        smoothed_x = max(0, min(smoothed_x, self.screen_width - 1))
        smoothed_y = max(0, min(smoothed_y, self.screen_height - 1))
        
        return smoothed_x, smoothed_y
    
    def detect_gestures(self, landmarks):
        gestures = {}

        #pinch to click or copy
        pinch_distance = self.calculate_distance(landmarks['thumb_tip'], landmarks['index_tip'])
        gestures['is_pinching'] = pinch_distance < self.pinch_threshold
        gestures['is_copying'] = gestures['is_pinching'] 

        #thumb and middle to scroll or paste
        scroll_distance = self.calculate_distance(landmarks['thumb_tip'], landmarks['middle_tip'])
        gestures['is_scrolling'] = scroll_distance < self.pinch_threshold
        gestures['is_pasting'] = gestures['is_scrolling'] 
 
        #index and ring to undo
        back_distance = self.calculate_distance(landmarks['thumb_tip'], landmarks['ring_tip'])
        gestures['is_backing'] = back_distance < self.pinch_threshold

        #index and pinky to toggle hotkey mode
        hot_distance = self.calculate_distance(landmarks['thumb_tip'], landmarks['pinky_tip'])
        gestures['is_hot_toggling'] = hot_distance < self.pinch_threshold

        #have to process this here and not in the other functions
        current_time = time.time()
        if gestures['is_hot_toggling'] and not self.was_hot_toggling:
            if current_time - self.last_hot_toggle_time > self.hot_toggle_cooldown:
                self.hot_mode = not self.hot_mode
                self.last_hot_toggle_time = current_time
        
        self.was_hot_toggling = gestures['is_hot_toggling']
        
        return gestures
    
    def update_status_display(self, frame, gestures):
        #displays the action text for debug
        pinch_text = "Pinching" if gestures['is_pinching'] else "Not Pinching"
        scroll_text = "Scrolling" if gestures['is_scrolling'] else "Not Scrolling"
        hot_text = f"HotKey mode: {'On' if self.hot_mode else 'Off'}"
        
        cv2.putText(frame, pinch_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, scroll_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, hot_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
    
    def handle_mouse_mode(self, smoothed_x, smoothed_y, gestures):
        current_time = time.time()
        #from gesutre detection method
        if gestures['is_scrolling']:
            if self.prev_scroll_y == 0:
                self.prev_scroll_y = smoothed_y
            else:
                scroll_amount = (self.prev_scroll_y - smoothed_y) * self.scroll_sensitivity
                #smooth scrolling 
                if abs(scroll_amount) > 0:
                    pag.scroll(int(scroll_amount))
                self.prev_scroll_y = smoothed_y
        else:
            self.prev_scroll_y = 0
            #otherwise just move the mouse
            pag.moveTo(smoothed_x, smoothed_y)

        if gestures['is_pinching'] and not self.was_pinching:
            if current_time - self.last_action_time > self.action_cooldown: #cooldown to not flicker
                pag.mouseDown()
                self.last_action_time = current_time
            self.was_pinching = True
        elif not gestures['is_pinching'] and self.was_pinching:
            pag.mouseUp()
            self.was_pinching = False
    
    def handle_hotkey_mode(self, gestures):
        current_time = time.time()
        #gestures for hotkeys, need to be in different method due to toggle
        
        if gestures['is_copying'] and not self.was_copying and current_time - self.last_action_time > self.action_cooldown:
            pag.hotkey('command', 'c')
            self.last_action_time = current_time
            self.was_copying = True
        elif not gestures['is_copying'] and self.was_copying:
            self.was_copying = False

        elif gestures['is_pasting'] and not self.was_pasting and current_time - self.last_action_time > self.action_cooldown:
            pag.hotkey('command', 'v')
            self.last_action_time = current_time
            self.was_pasting = True
        elif not gestures['is_pasting'] and self.was_pasting:
            self.was_pasting = False

        elif gestures['is_backing'] and not self.was_backing and current_time - self.last_action_time > self.action_cooldown:
            pag.hotkey('command', 'z')
            self.last_action_time = current_time
            self.was_backing = True
        elif not gestures['is_backing'] and self.was_backing:
            self.was_backing = False
    
    def handle_no_hand_detected(self, frame):
        #tells program what to do if hand is lost
        if self.was_pinching:
            pag.mouseUp()
            self.was_pinching = False
            self.prev_scroll_y = 0
        
        hot_text = f"HotKey mode: {'On' if self.hot_mode else 'Off'}"
        cv2.putText(frame, hot_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
    
    def run(self):
        #main webcam loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = self.process_frame(frame)
            cv2.imshow("Hand Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Hand tracking stopped.")


def main():
    controller = HandTrackingController()
    controller.run()

if __name__ == "__main__": #so i can import this file without running the main loop
    main()