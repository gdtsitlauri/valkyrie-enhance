from __future__ import annotations

from ..config import ValkyrieConfig
from ..types import ModuleDecision


class PerceptualAIEngineModule:
    name = "perceptual_ai_engine"

    def __init__(self, config: ValkyrieConfig) -> None:
        self.config = config
        self.preference_bias = dict(config.learning.preference_bias)

    def refine(self, decision: ModuleDecision) -> ModuleDecision:
        if self.preference_bias["stability"] > self.preference_bias["sharpness"]:
            decision.quality_multipliers["temporal_consistency"] = min(
                1.2,
                decision.quality_multipliers.get("temporal_consistency", 1.0) + 0.1,
            )
        if self.preference_bias["hdr"] < 0.4:
            decision.enabled["hdr_reconstruction"] = False
        return decision

    def learn_from_feedback(self, category: str, delta: float) -> None:
        if category not in self.preference_bias:
            return
        rate = self.config.learning.adaptation_rate
        self.preference_bias[category] = max(0.0, min(1.0, self.preference_bias[category] + delta * rate))
