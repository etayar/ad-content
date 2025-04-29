import os
import streamlit as st
import pandas as pd


# Function to list all video files in a directory
def list_video_files(directory):
    video_files = []
    for file in os.listdir(directory):
        if file.endswith((".mp4", ".avi", ".mov")):
            video_files.append(file)
    return video_files


# Main function that runs the labeling app
def main():
    st.title("Commercial Labeling Tool")  # Title for the web app

    video_dir = "data/raw"  # Directory where raw videos are stored
    label_file = "data/processed/labels.csv"  # Path to save labeled data

    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(label_file), exist_ok=True)

    # Load existing labels if they exist, otherwise create an empty DataFrame
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
    else:
        labels_df = pd.DataFrame(columns=["video_name", "industry", "audience", "sexual_content"])

    # List all video files
    video_files = list_video_files(video_dir)
    # Get set of already labeled video names
    labeled_videos = set(labels_df['video_name'])

    # Find videos that still need labeling
    videos_to_label = [vf for vf in video_files if vf not in labeled_videos]

    # If there are videos to label
    if videos_to_label:
        video = videos_to_label[0]  # Pick the first unlabeled video
        st.video(os.path.join(video_dir, video))  # Display the video

        # Predefined industry options for labeling (dropdown menu)
        industry_options = [
            "Automotive",
            "Food & Beverage",
            "Technology",
            "Fashion & Beauty",
            "Health & Pharma",
            "Retail",
            "Financial Services",
            "Travel & Hospitality",
            "Entertainment",
            "Toys & Games",
            "Nonprofit/PSA",
            "Other"
        ]

        # Create dropdown inputs for labeling
        industry = st.selectbox("Industry", industry_options)
        audience = st.selectbox("Audience", ["Kids", "Teens", "Adults", "Seniors", "All Ages"])
        sexual_content = st.selectbox("Sexual Content", ["Yes", "No"])

        # When the user clicks the submit button
        if st.button("Submit Label"):
            # Save the new label into the DataFrame
            new_label = {
                "video_name": video,
                "industry": industry,
                "audience": audience,
                "sexual_content": sexual_content
            }
            labels_df = labels_df.append(new_label, ignore_index=True)
            labels_df.to_csv(label_file, index=False)  # Save back to CSV
            st.success("Label saved! Reloading...")  # Show a success message
            st.experimental_rerun()  # Reload the app to show the next video
    else:
        st.write("All videos labeled!")  # If nothing left to label


# Entry point of the script
if __name__ == "__main__":
    main()
