import os
import streamlit as st
import pandas as pd

# Function to list all video files in a directory
def list_video_files(directory):
    return [f for f in os.listdir(directory) if f.endswith((".mp4", ".avi", ".mov"))]

# Main app
def main():
    st.title("Commercial Labeling Tool")

    video_dir = "data/raw"
    label_file = "data/processed/labels.csv"
    os.makedirs(os.path.dirname(label_file), exist_ok=True)

    # Load existing labels
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
    else:
        labels_df = pd.DataFrame(columns=["video_name", "industry", "audience", "family_friendly"])

    all_videos = list_video_files(video_dir)
    labeled_videos = set(labels_df["video_name"])
    videos_to_label = [vf for vf in all_videos if vf not in labeled_videos]

    mode = st.radio("Choose labeling mode:", ["Label new (unlabeled) videos", "Label any video (select manually)"])

    # --- Track index state for both modes ---
    if 'new_index' not in st.session_state:
        st.session_state.new_index = 0
    if 'manual_index' not in st.session_state:
        st.session_state.manual_index = 0

    if mode == "Label new (unlabeled) videos":
        if not videos_to_label:
            st.success("All videos are labeled!")
            return
        selected_video = videos_to_label[st.session_state.new_index]
    else:
        selected_video = st.selectbox(
            "Select a video to label (or re-label):",
            all_videos,
            index=st.session_state.manual_index,
            key="dropdown_video"
        )
        st.session_state.manual_index = all_videos.index(selected_video)

    st.video(os.path.join(video_dir, selected_video))

    # Fetch existing label
    existing_label = labels_df[labels_df["video_name"] == selected_video]
    if not existing_label.empty:
        industry = existing_label["industry"].values[0]
        audience = existing_label["audience"].values[0]
        family_friendly = existing_label["family_friendly"].values[0]
    else:
        industry = "Other"
        audience = "All Ages"
        family_friendly = "No"

    # Labeling inputs
    industry_options = [
        "Automotive", "Food & Beverage", "Technology", "Fashion & Beauty", "Health & Pharma", "Cosmetics & Toiletries",
        "Retail & E-commerce", "Financial Services", "Travel & Hospitality", "Entertainment & Media",
        "Toys & Children", "Nonprofit / PSA", "Delivery & Logistics", "Home & Furniture", "Other"
    ]

    industry = st.selectbox("Industry", industry_options, index=industry_options.index(industry))
    audience = st.selectbox("Audience", ["Kids", "Teens", "Adults", "Seniors", "All Ages"], index=["Kids", "Teens", "Adults", "Seniors", "All Ages"].index(audience))
    family_friendly = st.selectbox("Family-Friendly", ["Yes", "No"], index=["Yes", "No"].index(family_friendly))

    # Buttons
    if st.button("Submit Label"):
        new_label = {
            "video_name": selected_video,
            "industry": industry,
            "audience": audience,
            "family_friendly": family_friendly
        }
        labels_df = labels_df[labels_df["video_name"] != selected_video]
        labels_df = pd.concat([labels_df, pd.DataFrame([new_label])], ignore_index=True)
        labels_df.to_csv(label_file, index=False)
        st.success("Label saved (overwritten if existed).")
        if st.button("Back to Mode Selection"):
            st.session_state.clear()
            st.rerun()

    if st.button("Skip and Next"):
        if mode == "Label new (unlabeled) videos":
            st.session_state.new_index = (st.session_state.new_index + 1) % len(videos_to_label)
        else:
            st.session_state.manual_index = (st.session_state.manual_index + 1) % len(all_videos)
        st.rerun()

# Entry point
if __name__ == "__main__":
    main()
