import hydra
import numpy as np
import pytest
import torch
from hydra import compose, core, initialize

from romil.models import RoMIL, sine_embedding

torch.set_default_device("cpu")


@pytest.mark.parametrize(
    "model_name",
    ("clam_sb", "clam_mb", "RoPEAMIL_sb", "RoPEAMIL_mb", "RoPEDSMIL"),
)
@pytest.mark.parametrize("use_instance_loss", (True, False))
@pytest.mark.parametrize("apply_rotary", (True, False))
@pytest.mark.parametrize("absolute_position_embeddings", (True, False))
@pytest.mark.parametrize("batch_size", (2, 1, 10))
def test_models(
    model_name,
    use_instance_loss,
    batch_size,
    apply_rotary,
    absolute_position_embeddings,
):
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="test_data/")
    overrides = [
        f"model_name={model_name}",
        f"lightning_module.use_instance_loss={use_instance_loss}",
    ]
    if model_name.startswith("RoPEAMIL"):
        overrides.append(
            f"model_dict.{model_name}.positional_encoder.rope={apply_rotary}"
        )
        overrides.append(
            f"model_dict.{model_name}.absolute_position_embeddings={absolute_position_embeddings}"
        )

    cfg = compose(config_name="test_model_config", overrides=overrides)
    model = hydra.utils.instantiate(cfg["lightning_module"])
    optimizer = model.configure_optimizers()["optimizer"]

    if model_name == "RoPEDSMIL":
        batch_size = 1

    n_patches = torch.randint(10, 300, (batch_size,))

    test_tensor = [torch.randn((batch, 128)) for batch in n_patches]
    test_label = torch.randint(0, 1, (batch_size,))
    test_coords = [torch.randint(0, 300, (batch, 2)) for batch in n_patches]

    loss, instance_loss, preds, labels, probs = model.step(
        (test_tensor, test_label, test_coords)
    )

    loss.backward()
    optimizer.step()
    assert loss.item() > 0
    assert (instance_loss.item() > 0) == use_instance_loss or (
        isinstance(model.model, RoMIL.RoPEAMIL)
        or isinstance(model.model, RoMIL.RoPEDSMIL)
    )
    assert loss.size() == torch.Size([])
    assert instance_loss.size() == torch.Size([])
    assert preds.size() == torch.Size([batch_size])
    assert probs.size() == torch.Size([batch_size, 2])
    assert torch.allclose(probs.sum(-1), torch.ones((batch_size)))


@pytest.mark.parametrize("model_name", ("RoPEAMIL_sb", "RoPEAMIL_mb"))
@pytest.mark.parametrize("learnable_rotary", (False,))
def test_learnable_rotary(
    model_name,
    learnable_rotary,
):
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="test_data/")
    overrides = [
        f"model_name={model_name}",
        f"model_dict.{model_name}.positional_encoder.rope=True",
    ]

    cfg = compose(config_name="test_model_config", overrides=overrides)
    model = hydra.utils.instantiate(cfg["lightning_module"])
    optimizer = model.configure_optimizers()["optimizer"]

    n_patches = torch.randint(10, 300, (8,))

    test_tensor = [torch.randn((batch, 128)) for batch in n_patches]
    test_label = torch.randint(0, 1, (8,))
    test_coords = [torch.randint(0, 300, (batch, 2)) for batch in n_patches]

    init_rotary = torch.clone(
        model.model.positional_encoding.attention_blocks[0].attn.rope.inv_freq
    )

    loss, instance_loss, preds, labels, probs = model.step(
        (test_tensor, test_label, test_coords)
    )

    loss.backward()
    if learnable_rotary:
        assert not torch.all(
            model.model.positional_encoding.attention_blocks[0].attn.rope.inv_freq.grad
            == 0
        )
    else:
        assert (
            model.model.positional_encoding.attention_blocks[0].attn.rope.inv_freq.grad
            is None
        )
    optimizer.step()
    assert (not learnable_rotary) == np.array_equal(
        init_rotary.detach().numpy(),
        model.model.positional_encoding.attention_blocks[0]
        .attn.rope.inv_freq.detach()
        .numpy(),
    )


@pytest.mark.parametrize("use_instance_loss", (True, False))
def test_use_instance_loss(
    use_instance_loss,
):
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="test_data/")
    overrides = [
        f"model_name={'clam_mb'}",
        f"lightning_module.use_instance_loss={use_instance_loss}",
    ]

    cfg = compose(config_name="test_model_config", overrides=overrides)
    model = hydra.utils.instantiate(cfg["lightning_module"])
    optimizer = model.configure_optimizers()["optimizer"]

    n_patches = torch.randint(10, 300, (8,))

    test_tensor = [torch.randn((batch, 128)) for batch in n_patches]
    test_label = torch.randint(0, 1, (8,))
    test_coords = [torch.randint(0, 300, (batch, 2)) for batch in n_patches]

    loss, instance_loss, preds, labels, probs = model.step(
        (test_tensor, test_label, test_coords)
    )

    loss.backward()
    optimizer.step()

    assert (not use_instance_loss) == (instance_loss.grad_fn is None)


@pytest.mark.parametrize("model_name", ("RoPEAMIL_mb", "RoPEDSMIL"))
def test_embeddings_applied(model_name):
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="test_data/")

    overrides = [
        f"model_name={model_name}",
    ]

    batch_size = 8
    if model_name == "RoPEDSMIL":
        batch_size = 1

    n_patches = torch.randint(10, 300, (batch_size,))

    test_tensor = [torch.randn((batch, 128)) for batch in n_patches]
    test_coords = [torch.randint(0, 300, (batch, 2)) for batch in n_patches]

    outputs = []

    cfg = compose(config_name="test_model_config", overrides=overrides)
    model = hydra.utils.instantiate(cfg["lightning_module"])

    for position_embedding in [
        None,
        sine_embedding.SinePositionalEmbedding(
            cfg["model_dict"][model_name]["hidden_dim"]
        ),
    ]:
        model.model.absolute_position_embeddings = position_embedding
        outputs += [model.forward(features=test_tensor, coords=test_coords)["logits"]]
    for i in range(1, len(outputs)):
        assert not torch.allclose(outputs[0], outputs[i])


@pytest.mark.parametrize("model_name", ("RoPEAMIL_mb", "RoPEDSMIL"))
def test_inference_attention_weights(model_name):
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="test_data/")

    overrides = [
        f"model_name={model_name}",
    ]
    cfg = compose(config_name="test_model_config", overrides=overrides)
    model = hydra.utils.instantiate(cfg["lightning_module"])

    batch_size = 1

    n_patches = torch.randint(10, 300, (batch_size,))
    test_tensor = [torch.randn((batch, 128)) for batch in n_patches]
    test_coords = [torch.randint(0, 300, (batch, 2)) for batch in n_patches]

    model.model.mil_head.attention_net.output_inference_weights = True
    attn_weights = model.forward(features=test_tensor, coords=test_coords)[
        "attention_scores"
    ]

    assert not torch.allclose(attn_weights, torch.zeros_like(attn_weights))
    if cfg["model_name"].startswith("RoPEAMIL"):
        target_shape = torch.Size(
            (
                batch_size,
                model.model.n_classes,
                n_patches.sum(),
                cfg["lightning_module"]["model"]["mil_head"]["attention_net"]["n_head"],
            )
        )
    else:
        target_shape = torch.Size((batch_size, model.model.n_classes, n_patches.sum()))
    assert attn_weights.shape == target_shape


def test_dsmil_loss():
    core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="test_data/")
    model_name = "RoPEDSMIL"
    overrides = [
        f"model_name={model_name}",
    ]
    cfg = compose(config_name="test_model_config", overrides=overrides)
    model = hydra.utils.instantiate(cfg["lightning_module"])
    model.eval()
    batch_size = 1

    n_patches = torch.randint(10, 300, (batch_size,))
    test_tensor = [torch.randn((batch, 128)) for batch in n_patches]
    test_coords = [torch.randint(0, 300, (batch, 2)) for batch in n_patches]
    test_label = torch.ones((1)).to(int)
    loss, instance_loss, preds, labels, probs = model.step(
        (test_tensor, test_label, test_coords)
    )

    forward_outputs = model(test_tensor, test_coords)
    patch_loss = model.criterion(
        forward_outputs["patches_predictions"].max(1)[0], test_label
    )
    assert loss == model.bag_loss_weight * (
        0.5 * patch_loss + 0.5 * model.criterion(forward_outputs["logits"], test_label)
    )
