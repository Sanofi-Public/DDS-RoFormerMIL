import torch

from romil.models import attention


def test_same_output():
    """test custom sdpa has same attn_output as the pytorch implem"""
    n_head = 4
    n_tokens_total = 50
    n_classes = 8
    hidden_dim = 20

    k = torch.rand((1, n_head, n_tokens_total, hidden_dim))
    v = torch.rand((1, n_head, n_tokens_total, hidden_dim))
    q = torch.rand((1, n_head, n_classes, hidden_dim))

    class_attn_bias = torch.rand(1, n_head, n_classes, n_tokens_total) < 0.5
    class_attn_bias = torch.nan_to_num((class_attn_bias * -float("inf")), 0)

    gt = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=class_attn_bias,
    )

    (
        attention_outputs,
        attention_weights,
    ) = attention.inefficient_scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=class_attn_bias,
    )
    assert torch.allclose(attention_outputs, gt)


def test_specific_matrices():
    n_head = 4
    n_tokens_total = 50
    n_classes = 8
    hidden_dim = 20

    # Random tensor with a single attention target => should have attention weight=1
    k = torch.rand((1, n_head, n_tokens_total, hidden_dim))
    v = torch.rand((1, n_head, n_tokens_total, hidden_dim))
    q = torch.rand((1, n_head, n_classes, hidden_dim))

    target_id = torch.randint(0, n_tokens_total, (1,)).item()

    # everything to -inf except for one token
    class_attn_bias = torch.ones(1, n_head, n_classes, n_tokens_total)
    class_attn_bias[:, :, :, target_id] = 0
    class_attn_bias = torch.nan_to_num((class_attn_bias * -float("inf")), 0)
    output, weights = attention.inefficient_scaled_dot_product_attention(
        q, k, v, class_attn_bias
    )
    assert torch.equal(weights, (class_attn_bias > -1).to(int))
    assert torch.allclose(
        output, v[:, :, target_id, :][:, :, None, :].repeat(1, 1, 8, 1)
    )

    # All keys and values are similar -> all same attn_weights
    k = torch.ones((1, n_head, n_tokens_total, hidden_dim))
    v = torch.ones((1, n_head, n_tokens_total, hidden_dim))
    q = torch.ones((1, n_head, n_classes, hidden_dim))
    class_attn_bias = torch.zeros(1, n_head, n_classes, n_tokens_total)
    output, weights = attention.inefficient_scaled_dot_product_attention(
        q, k, v, class_attn_bias > 0
    )
    assert torch.all(weights == 1 / n_tokens_total)
    assert torch.allclose(output, torch.ones_like(output))
