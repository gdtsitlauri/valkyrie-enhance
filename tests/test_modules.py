import numpy as np

from valkyrie.config import ValkyrieConfig
from valkyrie.modules.audio import AudioEnhancementModule
from valkyrie.modules.scene import SceneDetectionModule
from valkyrie.modules.upscaling import UpscalingModule


def test_scene_detection_returns_valid_labels() -> None:
    frame = np.full((32, 32, 3), 127, dtype=np.uint8)
    scene = SceneDetectionModule().analyze(frame)
    assert scene.content_type in {"game", "film", "real_video"}
    assert scene.motion_level in {"static", "medium", "high"}


def test_audio_enhancement_preserves_shape() -> None:
    audio = np.random.randn(1024).astype(np.float32)
    enhanced = AudioEnhancementModule().process(audio)
    assert enhanced.shape == audio.shape


def test_upscaling_increases_resolution() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    upscaled = UpscalingModule(ValkyrieConfig()).process(frame, scale_factor=2.0)
    assert upscaled.shape[:2] == (32, 32)
