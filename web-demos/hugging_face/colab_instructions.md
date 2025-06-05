# Running ProPainter Demo in Google Colab

## Setup Instructions

1. **Create a New Colab Notebook**
   - Go to [Google Colab](https://colab.research.google.com)
   - Create a new notebook
   - Make sure to select GPU runtime:
     - Click `Runtime` → `Change runtime type`
     - Select `GPU` as Hardware accelerator
     - Click `Save`

2. **Clone and Setup Repository**
   ```python
   # Clone the repository
   !git clone https://github.com/sczhou/ProPainter.git
   
   # Change to the ProPainter directory
   %cd ProPainter
   
   # Create a symbolic link to make the model directory accessible
   !ln -s /content/ProPainter/model /content/ProPainter/web-demos/hugging_face/model
   ```

3. **Install Dependencies**
   ```python
   # Install basic requirements
   !pip install -r requirements.txt
   !pip install -r web-demos/hugging_face/requirements.txt
   !pip install torch torchvision
   
   # Install additional dependencies
   !pip install openmim
   !mim install mmcv-full
   !pip install mmdet
   !pip install mmsegmentation
   ```

4. **Download Required Models**
   ```python
   import os
   import sys
   
   # Add the project root to Python path
   current_dir = os.getcwd()
   sys.path.append(current_dir)
   sys.path.append(os.path.join(current_dir, 'model'))
   
   from utils.download_util import load_file_from_url
   
   # Create directories for models
   os.makedirs('weights', exist_ok=True)
   
   # Download SAM model
   sam_checkpoint = load_file_from_url(
       'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
       'weights'
   )
   
   # Download ProPainter model
   propainter_checkpoint = load_file_from_url(
       'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth',
       'weights'
   )
   ```

5. **Run the Demo**
   ```python
   # Make sure we're in the ProPainter directory
   %cd /content/ProPainter
   
   # Run the app with the correct Python path
   !PYTHONPATH=/content/ProPainter:/content/ProPainter/model python web-demos/hugging_face/app.py --port 7860 --share --device cuda
   ```

6. **Access the Demo**
   - After running the app, click on the public URL that appears in the output
   - The URL will look something like: `https://xxxx-xxxx-xxxx.gradio.live`

## Usage Guide

1. **Upload Video**
   - Click the upload button to select your video
   - Click "Get video info" to process the video

2. **Add Masks**
   - Use the slider to select a frame
   - Click on the image to add mask points
   - Click "Add mask" to confirm
   - You can add multiple masks if needed

3. **Tracking and Inpainting**
   - Click "Tracking" to track the masks
   - Adjust parameters if needed
   - Click "Inpainting" to process the video

## Troubleshooting

1. **If you get CUDA errors:**
   - Make sure you're using GPU runtime
   - Restart the runtime and try again

2. **If the app doesn't start:**
   - Check if port 7860 is available
   - Try changing the port number in the run command

3. **If models fail to download:**
   - Check your internet connection
   - Try running the download commands again

## Notes

- The demo requires a GPU to run efficiently
- Processing time depends on video length and resolution
- Keep the Colab tab open while processing
- You can download the processed video when it's done

## Memory Management

- If you get memory errors:
  - Reduce video resolution before uploading
  - Close other Colab tabs
  - Restart the runtime if needed

## Saving Results

- The processed video will be available for download
- You can also save it to your Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # Then copy your results to Drive
  !cp output_video.mp4 /content/drive/MyDrive/
  ``` 