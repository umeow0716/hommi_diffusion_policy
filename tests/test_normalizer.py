import torch

from hommi_diffusion_policy import LinearNormalizer


def test_round_trip_and_state_dict():
    data = {
        "obs": torch.randn(32, 4, 6),
        "action": torch.randn(32, 8, 3),
    }

    normalizer = LinearNormalizer()
    normalizer.fit(data)
    normalized = normalizer.normalize(data)
    restored = normalizer.unnormalize(normalized)

    assert torch.allclose(restored["obs"], data["obs"], atol=1e-5)
    assert torch.allclose(restored["action"], data["action"], atol=1e-5)

    clone = LinearNormalizer()
    clone.load_state_dict(normalizer.state_dict())
    clone_restored = clone.unnormalize(clone.normalize(data))
    assert torch.allclose(clone_restored["action"], data["action"], atol=1e-5)
