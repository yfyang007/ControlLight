from toolkit.extension import Extension


class AlphaLoraTrainerExtension(Extension):
    """LoRA trainer with alpha-conditioned current-batch visualization."""

    uid = "alpha_lora_trainer"
    name = "Alpha LoRA Trainer"

    @classmethod
    def get_process(cls):
        from .AlphaLoraTrainer import AlphaLoraTrainer

        return AlphaLoraTrainer


AI_TOOLKIT_EXTENSIONS = [
    AlphaLoraTrainerExtension,
]
