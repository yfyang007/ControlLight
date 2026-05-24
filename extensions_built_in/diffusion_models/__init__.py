AI_TOOLKIT_MODELS = []


def _try_register(import_func, name: str):
    try:
        models = import_func()
        AI_TOOLKIT_MODELS.extend(models)
    except Exception as e:
        print(f"Failed to import diffusion model group {name}. Error: {str(e)}")


def _chroma_models():
    from .chroma import ChromaModel, ChromaRadianceModel

    return [ChromaModel, ChromaRadianceModel]


def _hidream_models():
    from .hidream import HidreamModel, HidreamE1Model

    return [HidreamModel, HidreamE1Model]


def _f_light_models():
    from .f_light import FLiteModel

    return [FLiteModel]


def _omnigen2_models():
    from .omnigen2 import OmniGen2Model

    return [OmniGen2Model]


def _flux_kontext_models():
    from .flux_kontext import FluxKontextModel

    return [FluxKontextModel]


def _wan22_models():
    from .wan22 import Wan225bModel, Wan2214bModel, Wan2214bI2VModel

    return [Wan225bModel, Wan2214bI2VModel, Wan2214bModel]


def _qwen_image_models():
    from .qwen_image import QwenImageModel, QwenImageEditModel, QwenImageEditPlusModel

    return [QwenImageModel, QwenImageEditModel, QwenImageEditPlusModel]


def _flux2_models():
    from .flux2 import Flux2Model, Flux2Klein4BModel, Flux2Klein9BModel

    return [Flux2Model, Flux2Klein4BModel, Flux2Klein9BModel]


def _z_image_models():
    from .z_image import ZImageModel

    return [ZImageModel]


def _ltx2_models():
    from .ltx2 import LTX2Model, LTX23Model

    return [LTX2Model, LTX23Model]


def _zeta_chroma_models():
    from .zeta_chroma import ZetaChromaModel

    return [ZetaChromaModel]


def _ernie_image_models():
    from .ernie_image import ErnieImageModel

    return [ErnieImageModel]


def _nucleus_image_models():
    from .nucleus_image import NucleusImageModel

    return [NucleusImageModel]


# Keep FLUX.2 early and isolated: alpha-conditioned FLUX.2 Klein training should
# not fall back to StableDiffusion just because an unrelated optional model group
# has a missing dependency.
for _name, _import_func in [
    ("flux2", _flux2_models),
    ("chroma", _chroma_models),
    ("hidream", _hidream_models),
    ("f_light", _f_light_models),
    ("omnigen2", _omnigen2_models),
    ("flux_kontext", _flux_kontext_models),
    ("wan22", _wan22_models),
    ("qwen_image", _qwen_image_models),
    ("z_image", _z_image_models),
    ("ltx2", _ltx2_models),
    ("zeta_chroma", _zeta_chroma_models),
    ("ernie_image", _ernie_image_models),
    ("nucleus_image", _nucleus_image_models),
]:
    _try_register(_import_func, _name)
