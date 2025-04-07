import cv2
import numpy as np
import time
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText

def send_email(video_path):
    # Email setup
    from_email = "your_email@gmail.com"
    to_email = "recipient_email@gmail.com"
    subject = "Motion Detected - Video"
    body = "Motion was detected, and the video is attached."

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    # Attach the video file
    filename = os.path.basename(video_path)
    attachment = open(video_path, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {filename}")

    msg.attach(part)

    # Connect to Gmail SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    # Login to your email account
    server.login(from_email, "your_password_or_app_password")

    # Send the email
    server.send_message(msg)
    server.quit()
    print(f"Email sent with attachment {filename}.")

def send_email_async(video_path):
    # Run the email sending function in a separate thread
    email_thread = threading.Thread(target=send_email, args=(video_path,))
    email_thread.start()

def motion_detector(record_time=3):
    # Specify the folder where you want to save videos
    save_folder = 'recorded_videos'

    # Check if the folder exists, and if not, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    camera_index = 0  # Adjust this for your external camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Unable to open camera with index {camera_index}")
        return

    # Initialize the codec and create a VideoWriter object to save recordings
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 'H264' also works in some versions
    out = None
    is_recording = False
    record_start_time = None

    # Read the first frame to initialize the background
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Unable to read from camera")
        return

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale and blur it
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

        # Calculate the absolute difference between the current frame and the first frame
        frame_diff = cv2.absdiff(frame1_gray, frame2_gray)

        # Threshold the difference to get motion areas
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If motion is detected (any large contour is found)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1500:  # Adjust area threshold for sensitivity
                motion_detected = True
                break

        # Start recording if motion is detected
        if motion_detected and not is_recording:
            is_recording = True
            # Record start time in military format
            record_start_time = time.strftime("%Y%m%d_%H%M%S")
            
            # Save the video in the specified folder as MP4
            video_path = os.path.join(save_folder, f'motion_{record_start_time}.mp4')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame2.shape[1], frame2.shape[0]))
            print(f"Started recording: {video_path}")

        # If recording, write the current frame
        if is_recording:
            out.write(frame2)

            # Stop recording 
            if time.time() - time.mktime(time.strptime(record_start_time, "%Y%m%d_%H%M%S")) >= record_time:
                is_recording = False
                out.release()
                print(f"Recording saved as {video_path}")
                
                # Send the email with the recorded video asynchronously (in a separate thread)
                send_email_async(video_path)

        # Display the current frame
        cv2.imshow('Motion Detection', frame2)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the reference frame
        frame1_gray = frame2_gray.copy()

    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    motion_detector(record_time=3)
