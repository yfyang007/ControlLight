from toolkit.extension import Extension


class AlphaLoraEdgeLockTrainerExtension(Extension):
    """Alpha LoRA trainer with weighted FM masks and edge-locked structure loss."""

    uid = "alpha_lora_edge_lock_trainer"
    name = "Alpha LoRA Edge Lock Trainer"

    @classmethod
    def get_process(cls):
        from .AlphaLoraEdgeLockTrainer import AlphaLoraEdgeLockTrainer

        return AlphaLoraEdgeLockTrainer


AI_TOOLKIT_EXTENSIONS = [
    AlphaLoraEdgeLockTrainerExtension,
]
