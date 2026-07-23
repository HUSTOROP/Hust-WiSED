from .dataset import make_sib_diffusion_raw_field_dataset


SIB_DIFFUSION_RAW_FIELD_REGISTRY = {
    "sib_diffusion_raw_field": make_sib_diffusion_raw_field_dataset,
}
