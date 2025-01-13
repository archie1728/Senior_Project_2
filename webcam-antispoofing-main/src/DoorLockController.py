import RPi.GPIO as GPIO
import time
import logging
from threading import Lock
import json

class DoorLockController:
    def __init__(self, lock_pin, sensor_pin, emergency_pin):
        # Configure logging
        logging.basicConfig(filename='logs/door_lock.log', level=logging.INFO)
        self.logger = logging.getLogger('door_lock')
        
        # GPIO setup
        self.lock_pin = lock_pin
        self.sensor_pin = sensor_pin
        self.emergency_pin = emergency_pin
        self.lock = Lock()  # Thread safety
        
        # State tracking
        self.is_locked = True
        self.emergency_mode = False
        self.last_unlock_time = None
        self.unlock_count = 0
        
        # Safety parameters
        self.max_unlock_duration = 10  # seconds
        self.emergency_timeout = 300   # 5 minutes
        
        self.setup_gpio()
        self.setup_emergency_override()

    def setup_gpio(self):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.lock_pin, GPIO.OUT)
            GPIO.setup(self.sensor_pin, GPIO.IN)
            GPIO.setup(self.emergency_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.output(self.lock_pin, GPIO.HIGH)  # Default locked state
            
            self.logger.info("GPIO setup completed successfully")
        except Exception as e:
            self.logger.error(f"GPIO setup error: {str(e)}")
            raise

    def setup_emergency_override(self):
        try:
            GPIO.add_event_detect(
                self.emergency_pin, 
                GPIO.FALLING, 
                callback=self.emergency_override_callback, 
                bouncetime=300
            )
        except Exception as e:
            self.logger.error(f"Emergency override setup error: {str(e)}")
            raise

    def emergency_override_callback(self, channel):
        self.logger.warning("Emergency override activated")
        self.emergency_mode = True
        self.unlock_door(override=True)

    def unlock_door(self, duration=5, override=False):
        with self.lock:
            try:
                if self.emergency_mode and not override:
                    self.logger.warning("Unlock attempted during emergency mode")
                    return False

                GPIO.output(self.lock_pin, GPIO.LOW)  # Unlock
                self.is_locked = False
                self.last_unlock_time = time.time()
                self.unlock_count += 1
                
                self.logger.info(f"Door unlocked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if not override:
                    time.sleep(min(duration, self.max_unlock_duration))
                    self.lock_door()
                
                return True
            except Exception as e:
                self.logger.error(f"Door unlock error: {str(e)}")
                return False

    def lock_door(self):
        with self.lock:
            try:
                if self.emergency_mode:
                    return False

                GPIO.output(self.lock_pin, GPIO.HIGH)  # Lock
                self.is_locked = True
                self.logger.info(f"Door locked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                return True
            except Exception as e:
                self.logger.error(f"Door lock error: {str(e)}")
                return False

    def check_door_status(self):
        try:
            return {
                "is_locked": self.is_locked,
                "emergency_mode": self.emergency_mode,
                "last_unlock": self.last_unlock_time,
                "unlock_count": self.unlock_count,
                "sensor_status": GPIO.input(self.sensor_pin)
            }
        except Exception as e:
            self.logger.error(f"Status check error: {str(e)}")
            return None

    def save_status_log(self):
        try:
            status = self.check_door_status()
            with open('logs/door_status.json', 'w') as f:
                json.dump(status, f)
            self.logger.info("Status log saved successfully")
        except Exception as e:
            self.logger.error(f"Status log save error: {str(e)}")

    def reset_emergency_mode(self):
        with self.lock:
            self.emergency_mode = False
            self.lock_door()
            self.logger.info("Emergency mode reset")

    def cleanup(self):
        try:
            GPIO.cleanup([self.lock_pin, self.sensor_pin, self.emergency_pin])
            self.logger.info("GPIO cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

<antArtifact identifier="enhanced-anti-spoofing" type="application/vnd.ant.code" language="python" title="src/EnhancedAntiSpoofing.py">
import cv2
import numpy as np
from collections import deque
import logging
import time
from threading import Lock

class EnhancedAntiSpoofing:
    def __init__(self):
        # Configure logging
        logging.basicConfig(filename='logs/anti_spoofing.log', level=logging.INFO)
        self.logger = logging.getLogger('anti_spoofing')
        self.lock = Lock()

        # Motion detection parameters
        self.motion_threshold = 30
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        
        # Posture detection parameters
        self.head_tilt_threshold = 10
        self.posture_history = deque(maxlen=5)
        self.face_distance_threshold = 0.6  # Maximum allowed face distance
        
        # Safety timeouts
        self.last_detection_time = time.time()
        self.detection_timeout = 30  # seconds
        
        # Detection counters for analytics
        self.motion_detections = 0
        self.posture_violations = 0
        
        # Additional anti-spoofing parameters
        self.min_face_size = (30, 30)
        self.max_face_size = (300, 300)
        self.depth_check_enabled = False  # Enable if using depth camera

    def detect_motion(self, frame):
        with self.lock:
            try:
                if self.prev_frame is None:
                    self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(self.prev_frame, gray)
                self.prev_frame = gray

                # Calculate motion
                motion = np.mean(frame_diff) > self.motion_threshold
                self.motion_history.append(motion)
                
                if motion:
                    self.motion_detections += 1
                    self.logger.info(f"Motion detected at {time.strftime('%Y-%m-%d %H:%M:%S')}")

                return any(self.motion_history)
            except Exception as e:
                self.logger.error(f"Motion detection error: {str(e)}")
                return False

    def detect_posture(self, landmarks, face_location=None):
        with self.lock:
            try:
                if not landmarks:
                    return False

                # Basic posture checks
                nose = np.array(landmarks[0]['nose_bridge'])
                left_eye = np.mean(np.array(landmarks[0]['left_eye']), axis=0)
                right_eye = np.mean(np.array(landmarks[0]['right_eye']), axis=0)

                # Check head tilt
                eye_level_diff = abs(left_eye[1] - right_eye[1])
                head_tilt_ok = eye_level_diff < self.head_tilt_threshold

                # Check face orientation
                face_orientation_ok = self.check_face_orientation(nose, left_eye, right_eye)

                # Check face size if location provided
                face_size_ok = True
                if face_location:
                    face_size_ok = self.check_face_size(face_location)

                # Combined check
                posture_correct = all([head_tilt_ok, face_orientation_ok, face_size_ok])
                
                self.posture_history.append(posture_correct)
                
                if not posture_correct:
                    self.posture_violations += 1
                    self.logger.warning(f"Incorrect posture detected at {time.strftime('%Y-%m-%d %H:%M:%S')}")

                return all(self.posture_history)
            except Exception as e:
                self.logger.error(f"Posture detection error: {str(e)}")
                return False

    def check_face_orientation(self, nose, left_eye, right_eye):
        try:
            # Calculate face orientation based on facial landmarks
            eye_center = (left_eye + right_eye) / 2
            nose_tip = nose[-1]  # Last point of nose bridge
            
            # Check if face is looking straight ahead
            horizontal_diff = abs(eye_center[0] - nose_tip[0])
            vertical_diff = abs(eye_center[1] - nose_tip[1])
            
            return horizontal_diff < 20 and vertical_diff < 30
        except Exception as e:
            self.logger.error(f"Face orientation check error: {str(e)}")
            return False

    def check_face_size(self, face_location):
        try:
            top, right, bottom, left = face_location
            face_height = bottom - top
            face_width = right - left
            
            return (self.min_face_size[0] <= face_width <= self.max_face_size[0] and
                    self.min_face_size[1] <= face_height <= self.max_face_size[1])
        except Exception as e:
            self.logger.error(f"Face size check error: {str(e)}")
            return False

    def check_timeout(self):
        current_time = time.time()
        if current_time - self.last_detection_time > self.detection_timeout:
            self.logger.warning("Detection timeout occurred")
            return True
        return False

    def reset_detection(self):
        with self.lock:
            self.last_detection_time = time.time()
            self.motion_history.clear()
            self.posture_history.clear()
            self.prev_frame = None

    def get_statistics(self):
        return {
            "motion_detections": self.motion_detections,
            "posture_violations": self.posture_violations,
            "last_detection": self.last_detection_time
        }

    def update_thresholds(self, motion_threshold=None, head_tilt_threshold=None, 
                         face_distance_threshold=None):
        with self.lock:
            if motion_threshold is not None:
                self.motion_threshold = motion_threshold
            if head_tilt_threshold is not None:
                self.head_tilt_threshold = head_tilt_threshold
            if face_distance_threshold is not None:
                self.face_distance_threshold = face_distance_threshold
            
            self.logger.info("Thresholds updated")

<antArtifact identifier="system-tests" type="application/vnd.ant.code" language="python" title="tests/system_tests.py">
import unittest
import cv2
import numpy as np
import os
import sys
import time

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.EnhancedAntiSpoofing import EnhancedAntiSpoofing
from src.DoorLockController import DoorLockController

class TestAntiSpoofing(unittest.TestCase):
    def setUp(self):
        self.anti_spoofing = EnhancedAntiSpoofing()

    def test_motion_detection(self):
        # Create test frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:200, 100:200] = 255  # Add motion
        
        self.assertFalse(self.anti_spoofing.detect_motion(frame1))
        self.assertTrue(self.anti_spoofing.detect_motion(frame2))

    def test_posture_detection(self):
        # Test with simulated landmarks
        landmarks = [{
            'nose_bridge': [(100, 100), (100, 110)],
            'left_eye': [(90, 90), (95, 90)],
            'right_eye': [(110, 90), (115, 90)]
        }]
        
        self.assertTrue(self.anti_spoofing.detect_posture(landmarks))

    def test_face_size_check(self):
        # Test face size validation
        valid_face = (50, 100, 150, 200)  # Normal size
        too_small = (0, 10, 20, 30)      # Too small
        too_large = (0, 400, 400, 800)   # Too large
        
        self.assertTrue(self.anti_spoofing.check_face_size(valid_face))
        self.assertFalse(self.anti_spoofing.check_face_size(too_small))
        self.assertFalse(self.anti_spoofing.check_face_size(too_large))

class TestDoorLock(unittest.TestCase):
    def setUp(self):
        # Use test pins for unit testing
        self.door_controller = DoorLockController(18, 23, 24)

    def test_lock_unlock_cycle(self):
        self.assertTrue(self.door_controller.is_locked)
        self.assertTrue(self.door_controller.unlock_door())
        self.assertFalse(self.door_controller.is_locked)
        time.sleep(1)  # Wait for lock cycle
        self.assertTrue(self.door_controller.is_locked)

    def test_emergency_override(self):
        self.door_controller.emergency_override_callback(24)
        self.assertTrue(self.door_controller.emergency_mode)
        self.assertFalse(self.door_controller.is_locked)

    def test_status_check(self):
        status = self.door_controller.check_door_status()
        self.assertIsNotNone(status)
        self.assertIn('is_locked', status)
        self.assertIn('emergency_mode', status)

    def tearDown(self):
        self.door_controller.cleanup()

if __name__ == '__main__':
    unittest.main()