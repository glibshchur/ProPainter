"""
Stream processing components for handling longer videos.
"""

import os
import cv2
from PIL import Image
import scipy.ndimage
import numpy as np
import torch
from typing import Sequence

class Sequencer:
    """Base class for sequence processing."""
    def __init__(self, data):
        self.data = data
        self.buffer = None
        
    def __len__(self):
        return len(self.data)
        
    def _calc_data_items(self, raw_data_chunk_list):
        raise NotImplementedError()
        
    def _expand_buffer_by(self, data_chunk):
        raise NotImplementedError()

class BufferedSequencer(Sequencer):
    """Base class for buffered sequence processing."""
    def __init__(self, **kwargs):
        super(BufferedSequencer, self).__init__(**kwargs)
        
    def _calc_data_items(self, raw_data_chunk_list):
        raise NotImplementedError()
        
    def _expand_buffer_by(self, data_chunk):
        raise NotImplementedError()

class PillowImageRescaler:
    """Image rescaler using Pillow."""
    def __init__(self, image_resize_ratio: float):
        super(PillowImageRescaler, self).__init__()
        assert (image_resize_ratio > 0.0)
        self.image_resize_ratio = image_resize_ratio
        self.image_raw_size = None
        self.image_scaled_size = None
        self.do_scale = False

    def check_image_scale(self, image: np.ndarray):
        if self.image_raw_size is None:
            height, width = image.shape[:2]
            self.image_raw_size = (width, height)
            self.image_scaled_size = (
                int(self.image_resize_ratio * self.image_raw_size[0]),
                int(self.image_resize_ratio * self.image_raw_size[1])
            )
            self.image_scaled_size = (
                self.image_scaled_size[0] - self.image_scaled_size[0] % 8,
                self.image_scaled_size[1] - self.image_scaled_size[1] % 8
            )
            if self.image_raw_size != self.image_scaled_size:
                self.do_scale = True

    def __call__(self, image: np.ndarray, is_mask: bool) -> np.ndarray:
        self.check_image_scale(image)
        if self.do_scale:
            assert (not is_mask) or (len(image.shape) == 2)
            image = Image.fromarray(image)
            image = image.resize(
                size=self.image_scaled_size,
                resample=(Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
            )
            image = np.array(image)
        return image

    def invert(self, image: np.ndarray) -> np.ndarray:
        if self.do_scale:
            assert (len(image.shape) == 3)
            image = Image.fromarray(image)
            image = image.resize(
                size=self.image_raw_size,
                resample=Image.Resampling.BICUBIC
            )
            image = np.array(image)
        return image

class ScipyMaskDilator:
    """Mask dilator using scipy."""
    def __init__(self, dilation: int):
        super(ScipyMaskDilator, self).__init__()
        assert (dilation >= 0)
        self.dilation = dilation

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        if self.dilation > 0:
            mask = scipy.ndimage.binary_dilation(
                input=mask,
                iterations=self.dilation
            ).astype(np.uint8)
        else:
            assert (mask.dtype == np.uint8)
            assert (np.max(mask).item() <= 1)
        return mask

class FilePathDirSequencer(Sequencer):
    """Sequencer for file paths in directory."""
    def __init__(self, dir_path: str):
        assert os.path.exists(dir_path)
        self.dir_path = dir_path
        self.file_name_list = sorted(os.listdir(dir_path))
        super(FilePathDirSequencer, self).__init__(data=self.file_name_list)

    def __getitem__(self, index: int | slice) -> list[str]:
        selected_file_name_list = self.file_name_list[index]
        if isinstance(selected_file_name_list, str):
            return os.path.join(self.dir_path, selected_file_name_list)
        elif isinstance(selected_file_name_list, list):
            return [os.path.join(self.dir_path, x) for x in selected_file_name_list]
        else:
            raise ValueError()

class RawFrameSequencer(BufferedSequencer):
    """Sequencer for raw video frames."""
    def __init__(self, **kwargs):
        super(RawFrameSequencer, self).__init__(**kwargs)

    def load_frame(self, image_path: str) -> np.ndarray:
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _calc_data_items(self, raw_data_chunk_list):
        assert (len(raw_data_chunk_list) == 1)
        frames = raw_data_chunk_list[0]
        if isinstance(frames[0], str):
            frames = [self.load_frame(x) for x in frames]
        frames = np.stack(frames)
        return frames

    def _expand_buffer_by(self, data_chunk):
        if self.buffer is None:
            self.buffer = data_chunk
        else:
            self.buffer = np.concatenate([self.buffer, data_chunk])

class RawMaskSequencer(BufferedSequencer):
    """Sequencer for raw video masks."""
    def __init__(self, pre_raw_mask_dilation: int = 0, **kwargs):
        super(RawMaskSequencer, self).__init__(**kwargs)
        self.pre_raw_dilator = ScipyMaskDilator(dilation=pre_raw_mask_dilation)

    def load_mask(self, image_path: str) -> np.ndarray:
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        mask = self.pre_raw_dilator(mask)
        return mask

    def _calc_data_items(self, raw_data_chunk_list):
        assert (len(raw_data_chunk_list) == 1)
        masks = raw_data_chunk_list[0]
        if isinstance(masks[0], str):
            masks = [self.load_mask(x) for x in masks]
        masks = np.stack(masks)
        return masks

    def _expand_buffer_by(self, data_chunk):
        if self.buffer is None:
            self.buffer = data_chunk
        else:
            self.buffer = np.concatenate([self.buffer, data_chunk])

class FrameSequencer(BufferedSequencer):
    """Sequencer for processed video frames."""
    def __init__(self, image_resize_ratio: float, use_cuda: bool = True, **kwargs):
        super(FrameSequencer, self).__init__(**kwargs)
        self.rescaler = PillowImageRescaler(image_resize_ratio=image_resize_ratio)
        self.use_cuda = use_cuda

    def _calc_data_items(self, raw_data_chunk_list):
        assert (len(raw_data_chunk_list) == 1)
        frames = raw_data_chunk_list[0]
        assert (len(frames.shape) == 4)

        frames = np.array([self.rescaler(x, is_mask=False) for x in frames])
        frames = frames / 255.0 * 2 - 1.0

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = frames.float()

        if self.use_cuda:
            frames = frames.cuda()

        return frames

    def _expand_buffer_by(self, data_chunk):
        if self.buffer is None:
            self.buffer = data_chunk
        else:
            self.buffer = torch.cat([self.buffer, data_chunk])

class MaskSequencer(BufferedSequencer):
    """Sequencer for processed video masks."""
    def __init__(self, image_resize_ratio: float, mask_dilation: int, use_cuda: bool = True, **kwargs):
        super(MaskSequencer, self).__init__(**kwargs)
        self.rescaler = PillowImageRescaler(image_resize_ratio=image_resize_ratio)
        self.dilator = ScipyMaskDilator(dilation=mask_dilation)
        self.use_cuda = use_cuda

    def _calc_data_items(self, raw_data_chunk_list):
        assert (len(raw_data_chunk_list) == 1)
        masks = raw_data_chunk_list[0]
        assert (len(masks.shape) == 3)

        masks = np.array([self.dilator(self.rescaler(x, is_mask=True)) for x in masks])

        masks = np.expand_dims(masks, axis=-1)
        masks = torch.from_numpy(masks).permute(0, 3, 1, 2).contiguous()
        masks = masks.float()

        if self.use_cuda:
            masks = masks.cuda()

        return masks

    def _expand_buffer_by(self, data_chunk):
        if self.buffer is None:
            self.buffer = data_chunk
        else:
            self.buffer = torch.cat([self.buffer, data_chunk])

class ProPainterSIMSequencer(Sequencer):
    """Scaled Inpaint Masking (ProPainter-SIM) sequencer."""
    def __init__(self, inp_frames: Sequence, raw_frames: RawFrameSequencer,
                 raw_masks: RawMaskSequencer, rescaler: PillowImageRescaler,
                 post_raw_mask_dilation: int = 0):
        assert (len(raw_frames) > 0)
        super(ProPainterSIMSequencer, self).__init__(data=[inp_frames, raw_frames, raw_masks])
        self.rescaler = rescaler
        self.post_raw_dilator = ScipyMaskDilator(dilation=post_raw_mask_dilation)

    def _calc_data_items(self, raw_data_chunk_list):
        assert (len(raw_data_chunk_list) == 3)

        inp_frames = raw_data_chunk_list[0]
        raw_frames = raw_data_chunk_list[1]
        raw_masks = raw_data_chunk_list[2]

        assert isinstance(inp_frames, torch.Tensor)
        assert isinstance(raw_frames, np.ndarray)
        assert isinstance(raw_masks, np.ndarray)

        inp_frames_np = self.conv_propainter_frames_into_numpy(inp_frames)

        do_masking = (self.post_raw_dilator.dilation > 0)

        if self.rescaler.do_scale:
            assert (inp_frames_np.shape != raw_frames.shape)
            inp_frames_np = np.array([self.rescaler.invert(x) for x in inp_frames_np])
            do_masking = True

        if do_masking:
            dilated_raw_masks = np.array([self.post_raw_dilator(x) for x in raw_masks])
            dilated_raw_masks = np.expand_dims(dilated_raw_masks, axis=-1)
            inp_frames_np = inp_frames_np * dilated_raw_masks + raw_frames * (1 - dilated_raw_masks)

        return inp_frames_np

    @staticmethod
    def conv_propainter_frames_into_numpy(frames: torch.Tensor) -> np.ndarray:
        frames = (((frames + 1.0) / 2.0) * 255).to(torch.uint8)
        frames = frames.permute(0, 2, 3, 1).cpu().detach().numpy()
        return frames

class StreamingProPainterIterator:
    """Video Inpainting (ProPainter) iterator for streaming processing."""
    def __init__(self, raw_frames: RawFrameSequencer, raw_masks: RawMaskSequencer,
                 image_resize_ratio: float, mask_dilation: int,
                 post_raw_mask_dilation: int = 0, **kwargs):
        self.frames = FrameSequencer(
            data=raw_frames,
            image_resize_ratio=image_resize_ratio)
        self.masks = MaskSequencer(
            data=raw_masks,
            image_resize_ratio=image_resize_ratio,
            mask_dilation=mask_dilation)
        self.sclaed_inp_frame_sequencer = ProPainterSIMSequencer(
            inp_frames=None,  # Will be set during processing
            raw_frames=raw_frames,
            raw_masks=raw_masks,
            rescaler=self.frames.rescaler,
            post_raw_mask_dilation=post_raw_mask_dilation)
        self.kwargs = kwargs

    def process_chunk(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Process a chunk of frames."""
        # Get frames and masks for this chunk
        frames_chunk = self.frames.data[start_idx:end_idx]
        masks_chunk = self.masks.data[start_idx:end_idx]
        
        # Process the chunk using ProPainter
        processed_frames = self._process_frames(frames_chunk, masks_chunk)
        
        # Update the sequencer with processed frames
        self.sclaed_inp_frame_sequencer.inp_frames = processed_frames
        
        # Get the final output frames
        return self.sclaed_inp_frame_sequencer._calc_data_items([
            processed_frames,
            self.frames.data[start_idx:end_idx],
            self.masks.data[start_idx:end_idx]
        ])

    def _process_frames(self, frames: np.ndarray, masks: np.ndarray) -> torch.Tensor:
        """Process frames using ProPainter model."""
        # This method should be implemented with the actual ProPainter processing logic
        raise NotImplementedError()

def run_streaming_propainter(frames: np.ndarray, masks: np.ndarray,
                           chunk_size: int = 100, **kwargs) -> np.ndarray:
    """Run ProPainter in streaming mode."""
    # Create sequencers
    raw_frame_sequencer = RawFrameSequencer(data=frames)
    raw_mask_sequencer = RawMaskSequencer(data=masks)
    
    # Create iterator
    iterator = StreamingProPainterIterator(
        raw_frames=raw_frame_sequencer,
        raw_masks=raw_mask_sequencer,
        **kwargs
    )
    
    # Process video in chunks
    processed_frames = []
    for i in range(0, len(frames), chunk_size):
        end_idx = min(i + chunk_size, len(frames))
        chunk_frames = iterator.process_chunk(i, end_idx)
        processed_frames.append(chunk_frames)
    
    return np.concatenate(processed_frames, axis=0) 