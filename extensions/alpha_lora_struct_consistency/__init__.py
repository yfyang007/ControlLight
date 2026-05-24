from toolkit.extension import Extension


class AlphaLoraStructConsistencyTrainerExtension(Extension):
    """Alpha LoRA trainer with same-sample structural consistency loss."""

    uid = "alpha_lora_struct_consistency_trainer"
    name = "Alpha LoRA Struct Consistency Trainer"

    @classmethod
    def get_process(cls):
        from .AlphaLoraStructConsistencyTrainer import (
            AlphaLoraStructConsistencyTrainer,
        )

        return AlphaLoraStructConsistencyTrainer


AI_TOOLKIT_EXTENSIONS = [
    AlphaLoraStructConsistencyTrainerExtension,
]
