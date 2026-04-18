import cv2
import numpy as np

from valkyrie.engine import ValkyrieEngine
from valkyrie.io import process_video_file


def test_process_video_file(tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 32))
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((32, 32, 3), 40 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    summary = process_video_file(ValkyrieEngine(), input_path, output_path, max_frames=2)
    assert summary.frames_processed == 2
    assert output_path.exists()
