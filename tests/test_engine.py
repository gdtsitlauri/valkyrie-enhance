import numpy as np

from valkyrie.config import ValkyrieConfig
from valkyrie.engine import ValkyrieEngine


def demo_frame() -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 40
    frame[:, :, 1] = 80
    frame[:, :, 2] = 120
    return frame


def test_engine_runs_end_to_end() -> None:
    config = ValkyrieConfig()
    config.quality.scale_factor = 1.5
    engine = ValkyrieEngine(config)
    result = engine.process(demo_frame(), np.zeros(512, dtype=np.float32))
    assert result.frame.shape[0] >= 96
    assert result.frame.shape[1] >= 96
    assert result.audio is not None
    assert result.benchmark is not None


def test_preference_learning_updates_bias() -> None:
    engine = ValkyrieEngine()
    before = engine.perceptual.preference_bias["hdr"]
    engine.perceptual.learn_from_feedback("hdr", -1.0)
    after = engine.perceptual.preference_bias["hdr"]
    assert after < before
