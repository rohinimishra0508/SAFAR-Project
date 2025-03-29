import streamlit as st
import sys
print(sys.executable)
import cv2
print("OpenCV is working:", cv2.__version__)
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque, defaultdict
import logging
import re
from datetime import datetime
from sklearn.preprocessing import MaxAbsScaler
from scipy.spatial.distance import euclidean
import os
import base64

import googlemaps
import webbrowser
import subprocess
import json

import streamlit as st
import pandas as pd
import base64
import os

# Streamlit CODE
from streamlit.components.v1 import html
from geopy.geocoders import Nominatim
import requests
import urllib.parse
import pandas as pd
import base64



# Define the functions as provided

# Function to draw boxes and log bounding box information
className = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0, 0)):
    height, width, _ = frame.shape
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 += offset[0]
        y1 += offset[1]
        x2 += offset[0]
        y2 += offset[1]

        center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cat = int(categories[i]) if categories is not None else 0
        color = (0, 255, 0)  # You can modify this as per category
        id = int(identities[i]) if identities is not None else 0
        name = className[cat]  # Use className to get the object category name

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{id}:{name}"
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Log the bounding box information
        logging.info(f"Bounding Box: ID={id}, Category={name}, Coordinates=({x1}, {y1}),({x2}, {y2})")

    return frame

def extract_info(log_file_path):
    dates, times, ids, categories, top_left_x, top_left_y, bottom_right_x, bottom_right_y, frame_numbers = [[] for _ in range(9)]

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # Initialize variables to track frame number
    frame_number = 1
    previous_timestamp = None

    for line in lines:
        if "Bounding Box" in line:
            parts = line.strip().split(' ')
            date, time = parts[0], parts[1].replace(',','.')  # Split by ',' and take the first part
            info = parts[4:]

            box_id = info[0].split('=')[1][:-1]

            # Handle single or multiple word categories
            if ',' in info[1]:
                category = info[1].split('=')[1][:-1]
                category_info_index = 2
            else:
                category = info[1].split('=')[1]
                if ',' in info[2]:
                    category += ' ' + info[2].split(',')[0]
                    category_info_index = 3
                else:
                    category += ' ' + info[2]
                    category_info_index = 4

            # Extract coordinates
            tlx = info[category_info_index].split('=')[1][1:-1]
            tly = info[category_info_index + 1].split(',')[0][0:-1]
            brx = info[category_info_index + 1].split(',')[1][1:]
            bry = info[category_info_index + 2][0:-1]

            # Check if the timestamp has changed
            current_timestamp = (date, time)
            if current_timestamp != previous_timestamp:
                frame_number += 1
                previous_timestamp = current_timestamp

            dates.append(date)
            times.append(time)
            ids.append(box_id)
            categories.append(category)
            top_left_x.append(tlx)
            top_left_y.append(tly)
            bottom_right_x.append(brx)
            bottom_right_y.append(bry)
            frame_numbers.append(frame_number)
        
    data = {
        'Date': dates,
        'Time': times,
        'ID': ids,
        'Category': categories,
        'TLX': top_left_x,
        'TLY': top_left_y,
        'BRX': bottom_right_x,
        'BRY': bottom_right_y,
        'Frame Number': frame_numbers
    }

    df = pd.DataFrame(data)
    
    # Ensure consistent category classification based on the most frequent category
    most_common_category = df.groupby('ID')['Category'].agg(lambda x: x.mode()[0])
    df['Category'] = df['ID'].map(most_common_category)
    
    # Filter rows to include only specified categories
    valid_categories = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    df = df[df['Category'].isin(valid_categories)]
    
    df[['ID', 'TLX', 'TLY', 'BRX', 'BRY']] = df[['ID', 'TLX', 'TLY', 'BRX', 'BRY']].astype(int)

    return df

def create_trajectory(df):
    df['X'] = (df['TLX'] + df['BRX']) / 2
    df['Y'] = (df['TLY'] + df['BRY']) / 2

    df.sort_values(by=['ID', 'Date', 'Time'], inplace=True)

    trajectories = []
    for (ID, category), group in df.groupby(['ID', 'Category']):
        trajectory = {
            'ID': ID,
            'Category': category,
            'X': group['X'].tolist(),
            'Y': group['Y'].tolist(),
            'Time': group['Time'].tolist(),
            'Frame Number': group['Frame Number'].tolist()  # Include Frame Number
        }
        trajectories.append(trajectory)

    for trajectory in trajectories:
        distances = [euclidean((trajectory['X'][i], trajectory['Y'][i]), (trajectory['X'][i+1], trajectory['Y'][i+1]))
                     for i in range(len(trajectory['X'])-1)]

        times = [(pd.to_datetime(trajectory['Time'][i+1]) - pd.to_datetime(trajectory['Time'][i])).total_seconds()
                 for i in range(len(trajectory['Time'])-1)]

        velocities = [dist / time if time != 0 else 0 for dist, time in zip(distances, times)]

        trajectory['Speed'] = velocities
    
    return trajectories

def create_dataframe(log_file_path):
    df = extract_info(log_file_path)
    trajectories = create_trajectory(df)
    
    data = []
    for trajectory in trajectories:
        for i in range(len(trajectory['Time'])):
            if i < len(trajectory['Speed']):
                speed = trajectory['Speed'][i]
            else:
                speed = 0  # or some default value if there's no speed for the last point
            data.append({
                'ID': trajectory['ID'],
                'Category': trajectory['Category'],
                'Time': pd.to_datetime(trajectory['Time'][i]),
                'Position_X': trajectory['X'][i],
                'Position_Y': trajectory['Y'][i],
                'Speed': speed,
                'Frame Number': trajectory['Frame Number'][i]  # Include Frame Number
            })
    
    result_df = pd.DataFrame(data)
    return result_df

def generate_trajectory_txt(log_file_path, output_file_path):
    trajectories = create_trajectory(extract_info(log_file_path))
    
    with open(output_file_path, "w") as file:
        for trajectory in trajectories:
            file.write(f"ID: {trajectory['ID']}, Category: {trajectory['Category']}\n")
            
            for time, position, speed in zip(trajectory['Time'], zip(trajectory['X'], trajectory['Y']), trajectory['Speed']):
                x, y = position
                file.write(f"Time: {time}, Position: ({x},{y}), Speed: {speed}\n")
            
            file.write("\n")

def parse_trajectory_file(file_path):
    # Regular expression patterns
    id_category_pattern = re.compile(r'ID:\s*(\d+),\s*Category:\s*(\w+)')
    time_pattern = re.compile(r'Time:\s*([\d:.]+),\s*Position:\s*\(([\d.]+),([\d.]+)\),\s*Speed:\s*([\d.]+)')

    # Data structure to store parsed data
    data = defaultdict(lambda: {'category': '', 'entries': []})

    current_id = None

    with open(file_path, 'r') as file:
        for line in file:
            id_category_match = id_category_pattern.match(line)
            if id_category_match:
                current_id = int(id_category_match.group(1))
                category = id_category_match.group(2)
                data[current_id]['category'] = category
            else:
                time_match = time_pattern.match(line)
                if time_match:
                    time_str = time_match.group(1)
                    position = (float(time_match.group(2)), float(time_match.group(3)))
                    speed = float(time_match.group(4))
                    data[current_id]['entries'].append({'time': time_str, 'position': position, 'speed': speed})

    return data


def calculate_average_speed(entries):
    total_speed = sum(entry['speed'] for entry in entries)
    return total_speed / len(entries) if entries else 0

def calculate_total_distance(entries):
    total_distance = 0
    for i in range(1, len(entries)):
        x1, y1 = entries[i-1]['position']
        x2, y2 = entries[i]['position']
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        total_distance += distance
    return total_distance

def process_data(data):
    for object_id, object_data in data.items():
        entries = object_data['entries']
        average_speed = calculate_average_speed(entries)
        total_distance = calculate_total_distance(entries)
        print(f"ID: {object_id}, Category: {object_data['category']}")
        print(f"  Average Speed: {average_speed:.2f}")
        print(f"  Total Distance: {total_distance:.2f}")


def generate_statistics(data):
    # Create an empty list to store dictionaries
    data_list = []

    # Convert processed data to list of dictionaries
    for object_id, object_data in data.items():
        entries = object_data['entries']
        average_speed = calculate_average_speed(entries)
        total_distance = calculate_total_distance(entries)
        data_list.append({
            'ID': object_id, 
            'Category': object_data['category'], 
            'Average Speed': average_speed, 
            'Total Distance': total_distance
        })

    # Create DataFrame from the list of dictionaries
    statistics_df = pd.DataFrame(data_list)

    # Initialize MaxAbsScaler
    scaler = MaxAbsScaler()

    # Apply the scaler to 'Average Speed' and 'Total Distance'
    statistics_df[['Relative Speed', 'Relative Distance']] = scaler.fit_transform(statistics_df[['Average Speed', 'Total Distance']])

    # Rescale to the range of 0 to 100
    statistics_df['Relative Speed'] = (statistics_df['Relative Speed'] * 100)
    statistics_df['Relative Distance'] = (statistics_df['Relative Distance'] * 100)

    # Drop the original 'Average Speed' and 'Total Distance' columns
    #statistics_df.drop(columns=['Average Speed', 'Total Distance'], inplace=True)

    return statistics_df


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def detect_unsafe_situations(speed_df, proximity_threshold=500.0):
    unsafe_situations = []
    unsafe_pairs = set()

    for time in speed_df['Time'].unique():
        df_time = speed_df[speed_df['Time'] == time]
        positions = df_time[['Position_X', 'Position_Y']].values
        frame_numbers = df_time['Frame Number'].values  # Get the frame numbers for the current time

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                entity_1 = df_time.iloc[i]
                entity_2 = df_time.iloc[j]

                # Skip if both entities are persons
                if entity_1['Category'] == 'person' and entity_2['Category'] == 'person':
                    continue

                pair = frozenset([entity_1['ID'], entity_2['ID']])
                if pair not in unsafe_pairs:
                    dist = calculate_distance(positions[i][0], positions[i][1], positions[j][0], positions[j][1])
                    if dist < proximity_threshold:
                        unsafe_situations.append({
                            "Time": time,
                            "Frame Number": frame_numbers[i],  # Include Frame Number
                            "Entity_1_ID": entity_1['ID'],
                            "Entity_1_Category": entity_1['Category'],
                            "Entity_2_ID": entity_2['ID'],
                            "Entity_2_Category": entity_2['Category'],
                            "Distance": dist
                        })
                        unsafe_pairs.add(pair)

    return pd.DataFrame(unsafe_situations)


# Streamlit CODE
# Other parts of your code remain as you provided



# Streamlit CODE
# Other parts of your code remain as you provided



import streamlit as st

# Home page content
def home():
    # st.title("Home Page")
    # Title for the app
    st.markdown("<h1 style='color: green;'>Video Object Tracking and Statistics</h1>", unsafe_allow_html=True)

    # File uploader section
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"], key="file_uploader")

    ###########################################################################################

    # Google Maps API Key
    API_KEY = SECRET_KEY 

    # Initialize geolocator for reverse geocoding
    geolocator = Nominatim(user_agent="geoapi")

    # Function to get address from coordinates
    def get_address(lat, lng):
        location = geolocator.reverse((lat, lng), timeout=10)
        return location.address if location else "Unknown location"

    # CSS for custom styling
    st.markdown(
        """
        <style>
        .stApp { background-color: #1c1c1c; color: white; }
        .stButton > button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px; border: none; cursor: pointer; }
        .stButton > button:hover { background-color: red; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color: green;'>Location Picker with Google Maps</h1>", unsafe_allow_html=True)

    # Check if location is already selected
    if "selected_lat" not in st.session_state or "selected_lng" not in st.session_state:
        st.session_state["selected_lat"] = 28.6139  # Default latitude (New Delhi)
        st.session_state["selected_lng"] = 77.2090  # Default longitude (New Delhi)
        st.session_state["zoom_level"] = 12

    # Search bar for location
    search_query = st.text_input("Search for a location", "")

    # Button to search for the location
    if st.button("Search Location"):
        if search_query:
            # URL-encode the search query
            encoded_query = urllib.parse.quote(search_query)
            # Use Google Maps Places API to get location data
            places_url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={encoded_query}&key={API_KEY}"
            response = requests.get(places_url)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'OK' and data['results']:
                    location = data['results'][0]['geometry']['location']
                    st.session_state["selected_lat"] = location['lat']
                    st.session_state["selected_lng"] = location['lng']
                    st.session_state["zoom_level"] = 15
                else:
                    st.error("Location not found. Please try another search term.")
            else:
                st.error("Failed to fetch location data. Please try again.")

    # Button to show the map
    if st.button("Add Location"):
        st.session_state["zoom_level"] = 12  # Reset zoom level for selecting a new location

    # HTML & JavaScript for Google Maps with pin placement and selection
    map_html = f"""
    <div id="map" style="height: 500px; width: 100%;"></div>
    <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}&callback=initMap" async defer></script>
    <script>
        var map;
        var marker;

        function initMap() {{
            var center = {{lat: {st.session_state["selected_lat"]}, lng: {st.session_state["selected_lng"]}}};
            map = new google.maps.Map(document.getElementById('map'), {{
                center: center,
                zoom: {st.session_state["zoom_level"]},
            }});

            marker = new google.maps.Marker({{
                position: center,
                map: map
            }});

            map.addListener('click', function(event) {{
                placeMarker(event.latLng);
            }});
        }}

        function placeMarker(location) {{
            marker.setPosition(location);
            map.setCenter(location);
            map.setZoom(15);  // Zoom in to the selected location

            // Set hidden input fields for Streamlit to use
            const lat = location.lat();
            const lng = location.lng();
            Streamlit.setComponentValue({{'latitude': lat, 'longitude': lng}});
        }}
    </script>
    """

    # Display the map with the selected location and update latitude/longitude on click
    html(map_html, height=550)

    # Use hidden input to retrieve the selected coordinates
    if "latitude" in st.session_state and "longitude" in st.session_state:
        st.session_state["selected_lat"] = st.session_state["latitude"]
        st.session_state["selected_lng"] = st.session_state["longitude"]
        st.session_state["zoom_level"] = 15

    # Confirm location and display confirmation message
    if st.button("Confirm Location"):
        lat, lng = st.session_state["selected_lat"], st.session_state["selected_lng"]
        selected_address = get_address(lat, lng)
        st.session_state["selected_address"] = selected_address
        st.write("Location confirmed")  # Display only the confirmation text

    # Rest of your code for video processing and statistics...
    if uploaded_file is not None:
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Uploading... 0%")

        # Save the uploaded file to a temporary location
        with open("temp_video.mp4", "wb") as f:
            file_bytes = uploaded_file.getvalue()
            f.write(file_bytes)
            for percent_complete in range(100):
                progress_bar.progress(percent_complete + 1)
                status_text.text(f"Uploading... {percent_complete + 1}%")
            status_text.text("Upload complete!")

        # Button to get statistics
        if st.button("Get Statistics"):
            # Set up logging
            log_file_path = "temp_log.log"
            logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')

            # Load YOLOv9 model
            model = YOLO('yolov9c.pt')

            # Show status while ByteTrack is running
            status_text.text("ByteTrack Running...")

            # Track objects using the model
            results = model.track("temp_video.mp4", save=True, stream=True)

            # Get the total number of frames
            cap = cv2.VideoCapture("temp_video.mp4")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Process bounding boxes from the frames
            frame_count = 0
            for frame_id, result in enumerate(results):
                frame = result.orig_img
                bbox_xyxy = result.boxes.xyxy
                identities = result.boxes.id
                categories = result.boxes.cls

                # Draw boxes and log bounding box info
                processed_frame = draw_boxes(frame, bbox_xyxy, draw_trails=True, identities=identities, categories=categories)

                frame_count += 1
                progress_percent = min(1.0, (frame_count / total_frames))  # Ensure progress_percent is between 0.0 and 1.0
                progress_bar.progress(progress_percent)
                status_text.text(f"ByteTrack Running... {progress_percent * 100:.2f}%")

            # Ensure the progress bar reaches 100%
            progress_bar.progress(1.0)
            status_text.text("ByteTrack complete!")

            # Store the status text in session state
            st.session_state.status_text = "ByteTrack complete!"

            # Generate the trajectory file
            generate_trajectory_txt(log_file_path, "temp_trajectory.txt")

            # Parse trajectory data
            data = parse_trajectory_file("temp_trajectory.txt")

            # Generate statistics
            statistics_df = generate_statistics(data)
            speed_df = create_dataframe(log_file_path)
            total_unsafe_situations = detect_unsafe_situations(speed_df, proximity_threshold=1500.0)

            # Store the statistics dataframe in session state
            st.session_state.statistics_df = statistics_df

            # Calculate average speed of each category
            category_avg_speed = statistics_df.groupby('Category')['Relative Speed'].mean().reset_index()
            category_avg_speed.columns = ['Category', 'Average Relative Speed']

            # Store the category average speed in session state
            st.session_state.category_avg_speed = category_avg_speed

            # Store the total unsafe situations in session state
            st.session_state.total_unsafe_situations = total_unsafe_situations

    # Check if statistics dataframe is available in session state
    if 'statistics_df' in st.session_state:
        statistics_df = st.session_state.statistics_df
        category_avg_speed = st.session_state.category_avg_speed
        total_unsafe_situations = st.session_state.total_unsafe_situations

        # Display category counts
        st.markdown("<h2 style='color: green; text-align: center; font-family: Arial, sans-serif; font-weight: bold;'>Category Counts</h2>", unsafe_allow_html=True)
        category_counts_html = "".join([
            f"<p style='font-size: 24px; color: deeporange; font-weight: bold; text-align: center; margin: 10px 0;'>{category.capitalize()}: <span style='color: darkgreen;'>{count}</span></p>"
            for category, count in statistics_df['Category'].value_counts().to_dict().items()
        ])
        st.markdown(category_counts_html, unsafe_allow_html=True)

        # Display total unsafe situations
        st.markdown(f"<p style='font-size: 24px; color: deeporange; font-weight: bold; text-align: center; margin: 10px 0;'>Total Unsafe Situations: <span style='color: darkgreen;'>{len(total_unsafe_situations)}</span></p>", unsafe_allow_html=True)

        # Display total unsafe situations DataFrame
        st.markdown("<h2 style='color: green; text-align: center; font-family: Arial, sans-serif; font-weight: bold;'>Unsafe Situations</h2>", unsafe_allow_html=True)
        st.dataframe(total_unsafe_situations)

        # Function to create a download link for the DataFrame
        def create_download_link(df, filename="unsafe_situations.csv"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Unsafe Situations CSV</a>'
            return href

        # Provide the download link for the total unsafe situations DataFrame
        st.markdown(create_download_link(total_unsafe_situations), unsafe_allow_html=True)

        # Display average speed of each category
        st.markdown("<h2 style='color: green; text-align: center; font-family: Arial, sans-serif; font-weight: bold;'>Average Relative Speed by Category</h2>", unsafe_allow_html=True)
        avg_speed_html = "".join([
            f"<p style='font-size: 24px; color: deeporange; font-weight: bold; text-align: center; margin: 10px 0;'>{category.capitalize()}: <span style='color: darkgreen;'>{speed:.2f}</span></p>"
            for category, speed in category_avg_speed.values
        ])
        st.markdown(avg_speed_html, unsafe_allow_html=True)

        # Display the statistics table
        st.markdown("<h2 style='color: darkorange; text-align: center;'>Statistics Table</h2>", unsafe_allow_html=True)

        # Add filter feature
        filter_category = st.selectbox("Filter by Category", ["All"] + statistics_df['Category'].unique().tolist())
        filter_id = st.selectbox("Filter by ID", ["All"] + statistics_df['ID'].unique().tolist())
        filter_speed = st.slider("Filter by Relative Speed >", min_value=0.0, max_value=100.0, value=0.0)

        filtered_df = statistics_df
        if filter_category != "All":
            filtered_df = filtered_df[filtered_df['Category'] == filter_category]
        if filter_id != "All":
            filtered_df = filtered_df[filtered_df['ID'] == int(filter_id)]
        filtered_df = filtered_df[filtered_df['Relative Speed'] > filter_speed]

        # Format the table contents to display decimal values up to two decimal places
        filtered_df = filtered_df.copy()  # Create a copy to avoid SettingWithCopyWarning
        filtered_df['Relative Speed'] = filtered_df['Relative Speed'].apply(lambda x: f"{x:.2f}")
        filtered_df['Relative Distance'] = filtered_df['Relative Distance'].apply(lambda x: f"{x:.2f}")

        # Use Streamlit's dataframe for better alignment and styling
        st.dataframe(filtered_df.style.set_properties(**{
            'background-color': 'lightgreen',  # Restored light green background
            'color': 'black',
            'border-color': 'black',
            'width': '100%'
        }).set_table_styles([{
            'selector': 'thead th',
            'props': [('background-color', 'lightgreen'), ('font-weight', 'bold')]
        }]).hide(axis="index"), use_container_width=True)

        # Display the output video
        st.markdown("<h2 style='color: orange; text-align: center;'>Output Video</h2>", unsafe_allow_html=True)  # Changed to green
        st.video("/Users/DELL/Downloads/runs/detect/track/temp_video.mp4")

        # Function to create a green download button for the video
        def download_video_button(video_path):
            with open(video_path, "rb") as file:
                video_bytes = file.read()
            b64 = base64.b64encode(video_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="output_video.mp4"><button style="background-color: green; color: white; padding: 10px 20px; border: none; border-radius: 5px;">Download Output Video</button></a>'
            return href

        # Provide the download button
        video_path = "/Users/DELL/Downloads/runs/detect/track/temp_video.mp4"  # Ensure this path is correct 
        print(os.path.exists(video_path)) 
        st.markdown(download_video_button(video_path), unsafe_allow_html=True)

    # Check if status text is available in session state
    if 'status_text' in st.session_state:
        status_text = st.empty()
        status_text.text(st.session_state.status_text)

# Location picker page content
def Simulation():
    st.title("Simulation")
    # Google Maps API Key
    API_KEY = "AIzaSyCjgVanxWrkcUokFyEXH9eTGFZb2Njey_w"
    geolocator = Nominatim(user_agent="geoapi")

    # Function to get address from coordinates
    def get_address(lat, lng):
        location = geolocator.reverse((lat, lng), timeout=10)
        return location.address if location else "Unknown location"

    # CSS for custom styling
    st.markdown(
        """
        <style>
        .stApp { background-color: #1c1c1c; color: white; }
        .stButton > button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px; border: none; cursor: pointer; }
        .stButton > button:hover { background-color: red; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color: green;'>Pick location with Google Maps</h1>", unsafe_allow_html=True)

    # Check if location is already selected
    if "selected_lat" not in st.session_state or "selected_lng" not in st.session_state:
        st.session_state["selected_lat"] = 22.5726  # Default latitude (Kolkata)
        st.session_state["selected_lng"] = 88.3639  # Default longitude (Kolkata)
        st.session_state["zoom_level"] = 12

    # Search bar for location with a unique key
    search_query = st.text_input("Search for a location", key="search_query_input")

    # Button to search for the location
    if st.button("Search Location", key="search_location_button"):
        if search_query:
            # URL-encode the search query
            encoded_query = urllib.parse.quote(search_query)
            # Use Google Maps Places API to get location data
            places_url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={encoded_query}&key={API_KEY}"
            response = requests.get(places_url)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'OK' and data['results']:
                    location = data['results'][0]['geometry']['location']
                    st.session_state["selected_lat"] = location['lat']
                    st.session_state["selected_lng"] = location['lng']
                    st.session_state["zoom_level"] = 15
                else:
                    st.error("Location not found. Please try another search term.")
            else:
                st.error("Failed to fetch location data. Please try again.")

    # HTML & JavaScript for Google Maps with pin placement and selection
    map_html = f"""
    <div id="map" style="height: 500px; width: 100%;"></div>
    <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}&callback=initMap" async defer></script>
    <script>
        var map;
        var marker;

        function initMap() {{
            var center = {{lat: 22.5726, lng: 88.3639}}; 
            map = new google.maps.Map(document.getElementById('map'), {{
                center: center,
                zoom: {st.session_state["zoom_level"]},
                mapTypeControl: true,
                streetViewControl: true,  
                fullscreenControl: true
            }});

            marker = new google.maps.Marker({{
                position: center,
                map: map,
                draggable: true,
                title: "Drag me to adjust location!"
            }});

            map.addListener('click', function(event) {{
                placeMarker(event.latLng);
            }});
        }}

        function placeMarker(location) {{
            marker.setPosition(location);
            map.setCenter(location);
            map.setZoom(15);  // Zoom in to the selected location

            // Set hidden input fields for Streamlit to use
            const lat = location.lat();
            const lng = location.lng();
            Streamlit.setComponentValue({{'latitude': lat, 'longitude': lng}});
        }}
    </script>
    """

    # Display the map with the selected location and update latitude/longitude on click
    st.components.v1.html(map_html, height=550)

    # Use hidden input to retrieve the selected coordinates
    if "latitude" in st.session_state and "longitude" in st.session_state:
        st.session_state["selected_lat"] = st.session_state["latitude"]
        st.session_state["selected_lng"] = st.session_state["longitude"]
        st.session_state["zoom_level"] = 15
    
    # road parameters.. junction type: 1-4 way intersection / Crossroads;
    #                                  2-Y-junction
    #                                  3-T-junction
    #                                  4-Link
    road_data = [
        {"latitude": 22.55901288, "longitude": 88.37280838, "road_width": 4.3, "junction": 1}, 
        {"latitude": 22.55896509, "longitude": 88.37267502, "road_width": 4.39, "junction": 2}, 
        {"latitude": 22.55436338, "longitude": 88.36449842, "road_width": 4.42, "junction": 3},
        {"latitude": 22.55234629, "longitude": 88.36374439, "road_width": 4.32, "junction": 1}, 
        {"latitude": 22.59421487, "longitude": 88.38613115, "road_width": 4.53, "junction": 4},   
        {"latitude": 22.58462783, "longitude": 88.37665835, "road_width": 4.4, "junction": 1}, 
        {"latitude": 22.57848539, "longitude": 88.39036915, "road_width": 4.29, "junction": 4}, 
        {"latitude": 22.58435622, "longitude": 88.38937364, "road_width": 3.79, "junction": 3},
        {"latitude": 22.53507843, "longitude": 88.30004794, "road_width": 4.4, "junction": 3}, 
        {"latitude": 22.53839509, "longitude": 88.30412348, "road_width": 4.39, "junction": 4},
        {"latitude": 22.52257585, "longitude": 88.30508469, "road_width": 4.42, "junction": 4},
        {"latitude": 22.51780841, "longitude": 88.30376255, "road_width": 4.42, "junction": 4},
        {"latitude": 22.61517244, "longitude": 88.37873506, "road_width": 4.22, "junction": 4},
        {"latitude": 22.59872179, "longitude": 88.36449842, "road_width": 3.8, "junction": 1},
        {"latitude": 22.57435217, "longitude": 88.36449842, "road_width": 4.31, "junction": 1},
        {"latitude": 22.56035445, "longitude": 88.36449842, "road_width": 4.42, "junction": 3},
        {"latitude": 22.54552164, "longitude": 88.36449842, "road_width": 4.42, "junction": 3},
        {"latitude": 22.53933834, "longitude": 88.36449842, "road_width": 4.36, "junction": 4},
        {"latitude": 22.5410245, "longitude": 88.36449842, "road_width": 4.35, "junction": 1},
        {"latitude": 22.53979303, "longitude": 88.37090981, "road_width": 4.33, "junction": 2},
        {"latitude": 22.5483029, "longitude": 88.37092404, "road_width": 4.41, "junction": 1},
        {"latitude": 22.57104413, "longitude": 88.34374724, "road_width": 4.41, "junction": 4},
        {"latitude": 22.57843417, "longitude": 88.34792041, "road_width": 4.31, "junction": 4},
        {"latitude": 22.53553198, "longitude": 88.34616928, "road_width": 4.3, "junction": 1},
        {"latitude": 22.50161539, "longitude": 88.32208021, "road_width": 3.65, "junction": 4},
        {"latitude": 22.50664331, "longitude": 88.31986371, "road_width": 4.31, "junction": 1},
        {"latitude": 22.48864924, "longitude": 88.27337305, "road_width": 4.38, "junction": 4},
        {"latitude": 22.51514256, "longitude": 88.40168721, "road_width": 3.47, "junction": 3},
        {"latitude": 22.47534995, "longitude": 88.31304885, "road_width": 4.31, "junction": 1},
        {"latitude": 22.45510984, "longitude": 88.30573401, "road_width": 4.38, "junction": 1},
        {"latitude": 22.49487439, "longitude": 88.31924214, "road_width": 4.36, "junction": 1},
        {"latitude": 22.49234887, "longitude": 88.39685871, "road_width": 4.38, "junction": 1}
    ]

    # Mapping junction numbers to readable names
    junction_mapping = {
        1: "4-way intersection / Crossroads",
        2: "Y-junction",
        3: "T-junction",
        4: "Link"
    }
    
    st.subheader("Enter Location Coordinates")
    longitude = st.number_input("Longitude", value=st.session_state["selected_lng"], format="%.8f")
    latitude = st.number_input("Latitude", value=st.session_state["selected_lat"], format="%.8f")

    # Confirm location button
    if st.button("Confirm Location", key="confirm_location_button"):
        st.session_state["selected_lat"] = latitude
        st.session_state["selected_lng"] = longitude
        st.session_state["zoom_level"] = 15
        st.write(f"Location confirmed: Latitude = {latitude}, Longitude = {longitude}")
    

    # Find matching road data
    matched_data = next((data for data in road_data if data["latitude"] == latitude and data["longitude"] == longitude), None)

    # If match found, update road width and junction type, else set to None
    if matched_data:
        road_width = matched_data["road_width"]
        junction_type = junction_mapping.get(matched_data["junction"], "None")
    else:
        road_width = 0.0
        junction_type = "None"
     
    # Panel for Traffic Inputs
    # Initialize session state variables
    if "show_panel" not in st.session_state:
        st.session_state["show_panel"] = False
    if "form_submitted" not in st.session_state:
        st.session_state["form_submitted"] = False
    if "traffic_data" not in st.session_state:  # ✅ Ensure traffic_data is initialized
        st.session_state["traffic_data"] = []

    # Function to toggle the panel
    def toggle_panel():
        st.session_state["show_panel"] = not st.session_state["show_panel"]
        st.session_state["form_submitted"] = False  # Reset form state

    # Button to show/hide the panel
    st.button("Enter Traffic Details", on_click=toggle_panel)

    # Display success message if form was submitted
    if st.session_state["form_submitted"]:
        st.success("Traffic details saved successfully! ✅")

    # Display the panel only if toggled and not yet submitted
    if st.session_state["show_panel"] and not st.session_state["form_submitted"]:
        st.subheader("Traffic Details Input Panel 🚦")

        with st.form("traffic_form"):
            st.write("### Vehicle Speeds (km/h)")
            car_speed = st.number_input("Car Speed", min_value=0.0, step=0.1)
            lgv_speed = st.number_input("LGV Speed", min_value=0.0, step=0.1)
            hgv_speed = st.number_input("HGV Speed", min_value=0.0, step=0.1)
            bus_speed = st.number_input("Bus Speed", min_value=0.0, step=0.1)
            man_speed = st.number_input("Manual Speed (Walking)", min_value=0.0, step=0.1)
            bike_man_speed = st.number_input("Bike Manual Speed", min_value=0.0, step=0.1)

            st.write("### Relative Flow")
            car_relative_flow = st.number_input("Car Relative Flow", min_value=0.0, step=0.01)
            lgv_relative_flow = st.number_input("LGV Relative Flow", min_value=0.0, step=0.01)
            hgv_relative_flow = st.number_input("HGV Relative Flow", min_value=0.0, step=0.01)
            man_relative_flow = st.number_input("Manual Relative Flow", min_value=0.0, step=0.01)
            bike_man_relative_flow = st.number_input("Bike Manual Relative Flow", min_value=0.0, step=0.01)

            # st.write("### Road Parameters")
            # road_width = st.number_input("Road Width (m)", min_value=0.0, step=0.1)

            # Custom button styling
            st.markdown(
                """
                <style>
                    div[data-testid="stForm"] button {
                        background-color: #28a745 !important; /* Green */
                        color: white !important;
                        font-size: 16px;
                        padding: 10px 20px;
                        border-radius: 5px;
                        border: none;
                    }
                    div[data-testid="stForm"] button:hover {
                        background-color: #218838 !important; /* Darker Green */
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Submit button
            submitted = st.form_submit_button("Save Traffic Data")
            if submitted:
                traffic_details = [car_speed, lgv_speed, hgv_speed, bus_speed, man_speed, bike_man_speed, 
                                    car_relative_flow, lgv_relative_flow, hgv_relative_flow,
                                    man_relative_flow, bike_man_relative_flow, road_width  
                                  ]
                
                # Append to session state array
                st.session_state["traffic_data"].append(traffic_details)  # ✅ Store in session state
                st.session_state["form_submitted"] = True
                st.session_state["show_panel"] = False  # Hide panel after submission
    # Display saved traffic data
    if st.session_state["traffic_data"]:  # ✅ No more KeyError
        st.write("### Saved Traffic Data:")
        for idx, data in enumerate(st.session_state["traffic_data"], start=1):
            st.write(f"**Entry {idx}:**", data)


# GIS Software Page
def GIS_Software():
    st.title("GIS Software")
    # Path to the CSV file
    csv_file_path = '/Users/ritwikghosh/Downloads/IIT Kgp SAFAR Work/Filtering Risky Analysis/data_display.csv'

    # Add a button to download the CSV file
    if st.button("Download Data Display CSV"):
        st.markdown(create_download_link(csv_file_path), unsafe_allow_html=True)

    # Load the CSV file
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

    # Display the table header
    st.markdown("<h2 style='color: darkorange; text-align: center;'>Data Display Table</h2>", unsafe_allow_html=True)

    # Load the CSV file (corrected path)
    csv_path = "/Users/ritwikghosh/Downloads/IIT Kgp SAFAR Work/Filtering Risky Analysis/data_display.csv"
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

    # Add filter features
    filter_column = st.selectbox("Filter by Column", ["All"] + df.columns.tolist())
    filter_value = st.text_input(f"Filter {filter_column} contains:") if filter_column != "All" else ""

    # Filter dataframe
    filtered_df = df[df[filter_column].astype(str).str.contains(filter_value, case=False)] if filter_column != "All" and filter_value else df

    # Display the dataframe with styling
    st.dataframe(filtered_df.style.set_properties(**{
        'background-color': 'lightgreen',
        'color': 'black',
        'border-color': 'black',
        'width': '100%'
    }).set_table_styles([{
        'selector': 'thead th',
        'props': [('background-color', 'lightgreen'), ('font-weight', 'bold')]
    }]).hide(axis="index"), use_container_width=True)

    # View GIS Map button logic
    if st.button("View GIS Map") and filter_value:
        # Mapping from filter column to folder paths
        folder_mapping = {
            "T.P.Guard": "'/Users/ritwikghosh/Downloads/IIT Kgp SAFAR Work/Filtering Risky Analysis/TP Gard Wise GIS file'",
            "Vehicle": "/Users/ritwikghosh/Downloads/IIT Kgp SAFAR Work/Filtering Risky Analysis/Offending Vehicle Wise GIS File",
            "Road User Type": "/Users/ritwikghosh/Downloads/IIT Kgp SAFAR Work/Filtering Risky Analysis/Road User Type(Victims) wise GIS Screenshot file"
        }

        # Get folder path based on selected column
        folder_path = folder_mapping.get(filter_column, "")
        
        # Construct image path if folder path exists
        if folder_path:
            # Normalize filter_value to avoid issues with case sensitivity or extra spaces
            normalized_filter_value = filter_value.strip()
            image_path = os.path.join(folder_path, f"{normalized_filter_value}.png")
            st.write(f"Attempting to load image from: {image_path}")
            
            # Check if the image exists before displaying it
            if os.path.exists(image_path):
                st.markdown("<h2 style='color: darkorange; text-align: center;'>GIS Map</h2>", unsafe_allow_html=True)
                st.image(image_path, caption=f"GIS Map for {normalized_filter_value}", use_container_width=True)
            else:
                st.warning(f"No image found for '{normalized_filter_value}' in the selected category.")
        else:
            st.warning("Please select a valid filter column.")


    if 'status_text' in st.session_state:
        status_text = st.empty()
        status_text.text(st.session_state.status_text)


# Create a selectbox for navigation
page = st.selectbox("Select a page", ("Home", "Simulation", "GIS Software"))

# Display the selected page's content
if page == "Home":
    home()
elif page == "Simulation":
    Simulation()
elif page == "GIS Software":
    GIS_Software()
