import torch

from hommi_diffusion_policy.model import (
    ActionDiT,
    ConditionalUnet1D,
    TransformerForActionDiffusion,
)


def test_unet_forward():
    model = ConditionalUnet1D(
        input_dim=3,
        global_cond_dim=8,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=8,
    )
    out = model(torch.randn(2, 8, 3), torch.tensor([1, 2]), global_cond=torch.randn(2, 8))
    assert out.shape == (2, 8, 3)


def test_transformer_forward():
    model = TransformerForActionDiffusion(
        input_dim=3,
        output_dim=3,
        action_horizon=8,
        n_layer=2,
        n_head=4,
        n_emb=16,
        max_cond_tokens=5,
    )
    out = model(
        torch.randn(2, 8, 3),
        torch.tensor([1, 2]),
        cond=torch.randn(2, 4, 16),
    )
    assert out.shape == (2, 8, 3)


def test_dit_forward():
    model = ActionDiT(
        obs_embed_dim=8,
        action_dim=3,
        action_len=8,
        embed_dim=16,
        timestep_embed_dim=16,
        depth=2,
        num_heads=4,
    )
    out = model(
        torch.randn(2, 8),
        torch.randn(2, 8, 3),
        torch.tensor([1, 2]),
    )
    assert out.shape == (2, 8, 3)


def test_transformer_decoder_layers_get_independent_attention_initialization():
    torch.manual_seed(0)
    model = TransformerForActionDiffusion(
        input_dim=3,
        output_dim=3,
        action_horizon=8,
        n_layer=3,
        n_head=4,
        n_emb=16,
        max_cond_tokens=5,
    )
    qkv0 = model.decoder.layers[0].self_attn.in_proj_weight
    qkv1 = model.decoder.layers[1].self_attn.in_proj_weight
    cross0 = model.decoder.layers[0].multihead_attn.in_proj_weight
    cross1 = model.decoder.layers[1].multihead_attn.in_proj_weight
    assert not torch.equal(qkv0, qkv1)
    assert not torch.equal(cross0, cross1)


def test_dit_uses_hommi_checkpoint_key_names():
    model = ActionDiT(
        obs_embed_dim=8,
        action_dim=3,
        action_len=8,
        embed_dim=16,
        timestep_embed_dim=16,
        depth=2,
        num_heads=4,
    )
    keys = set(model.state_dict())
    assert "dit_blocks.0.ada_ln_modulation.1.weight" in keys
    assert "head.ada_ln_modulation.1.weight" in keys
    assert "head.final_linear.weight" in keys
    assert not any(key.startswith("blocks.") for key in keys)


def test_unet_local_conditioning_matches_umi_effective_path():
    model = ConditionalUnet1D(
        input_dim=3,
        local_cond_dim=4,
        global_cond_dim=8,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=8,
    )
    out = model(
        torch.randn(2, 8, 3),
        torch.tensor([1, 2]),
        local_cond=torch.randn(2, 8, 4),
        global_cond=torch.randn(2, 8),
    )
    assert out.shape == (2, 8, 3)
    # The second module is intentionally retained for checkpoint compatibility.
    assert "local_cond_encoder.1.blocks.0.block.0.weight" in model.state_dict()


def test_unet_rejects_horizon_not_divisible_by_downsample_factor():
    model = ConditionalUnet1D(
        input_dim=3,
        global_cond_dim=8,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32, 64),
        n_groups=8,
    )
    with torch.no_grad():
        try:
            model(
                torch.randn(2, 15, 3),
                torch.tensor([1, 2]),
                global_cond=torch.randn(2, 8),
            )
        except ValueError as exc:
            assert "downsample factor 4" in str(exc)
        else:
            raise AssertionError("expected invalid U-Net horizon to be rejected")


def test_unet_final_groupnorm_matches_umi_default_groups():
    model = ConditionalUnet1D(
        input_dim=3,
        global_cond_dim=8,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=4,
    )
    assert model.final_conv[0].block[1].num_groups == 8
