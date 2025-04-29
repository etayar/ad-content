# ad-content

This project aims to build an ML pipeline that classifies commercials (advertisements) into categories such as:

- Product type
- Industry/domain
- Sexual content presence
- Intended audience

## Project Structure

- `data/`: Commercial videos and metadata
- `notebooks/`: Experiments and EDA
- `scripts/`: Data downloading, preprocessing, training scripts
- `models/`: Saved models and checkpoints

## Goals

- Multilabel classification (industry, audience, sexual content)
- Multimodal models (text, video frames, audio, etc.)

## Setup

```bash
pip install -r requirements.txt
```

## Additional System Requirements

This project requires `ffmpeg` to be installed for handling video downloads (used by yt-dlp).

- On MacOS:
  ```bash
  brew install ffmpeg
