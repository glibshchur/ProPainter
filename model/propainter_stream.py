"""
ProPainter streaming implementation for handling longer videos.
"""

import torch
import numpy as np
from model.stream import (StreamingProPainterIterator, RawFrameSequencer,
                         RawMaskSequencer)

class ProPainterStreamingIterator(StreamingProPainterIterator):
    """ProPainter streaming iterator implementation."""
    def __init__(self, raw_frames: RawFrameSequencer, raw_masks: RawMaskSequencer,
                 image_resize_ratio: float, mask_dilation: int,
                 post_raw_mask_dilation: int = 0, model=None, **kwargs):
        super().__init__(raw_frames=raw_frames, raw_masks=raw_masks,
                        image_resize_ratio=image_resize_ratio,
                        mask_dilation=mask_dilation,
                        post_raw_mask_dilation=post_raw_mask_dilation,
                        **kwargs)
        self.model = model

    def _process_frames(self, frames: np.ndarray, masks: np.ndarray) -> torch.Tensor:
        """Process frames using ProPainter model."""
        # Convert frames and masks to the format expected by ProPainter
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = frames.float() / 255.0 * 2 - 1.0
        
        masks = torch.from_numpy(masks).unsqueeze(1).float()
        
        if torch.cuda.is_available():
            frames = frames.cuda()
            masks = masks.cuda()
        
        # Process the frames using ProPainter
        with torch.no_grad():
            output_frames = self.model.baseinpainter.forward_sequence(
                frames, masks,
                neighbor_stride=self.kwargs.get('ref_stride', 5),
                neighbor_len=self.kwargs.get('neighbor_length', 10),
                subvideo_len=self.kwargs.get('subvideo_length', 80)
            )
        
        return output_frames

def run_streaming_inpainting(frames: np.ndarray, masks: np.ndarray,
                           model=None, chunk_size: int = 100, **kwargs) -> np.ndarray:
    """Run ProPainter inpainting in streaming mode."""
    # Create sequencers
    raw_frame_sequencer = RawFrameSequencer(data=frames)
    raw_mask_sequencer = RawMaskSequencer(data=masks)
    
    # Create iterator
    iterator = ProPainterStreamingIterator(
        raw_frames=raw_frame_sequencer,
        raw_masks=raw_mask_sequencer,
        model=model,
        **kwargs
    )
    
    # Process video in chunks
    processed_frames = []
    for i in range(0, len(frames), chunk_size):
        end_idx = min(i + chunk_size, len(frames))
        chunk_frames = iterator.process_chunk(i, end_idx)
        processed_frames.append(chunk_frames)
    
    return np.concatenate(processed_frames, axis=0) 