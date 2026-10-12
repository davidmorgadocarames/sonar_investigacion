"""Transformer and hybrid tests (F5-T04, F5-T05) on toy data, on CPU."""

import pytest
import torch
from test_cv import engine
from torch import nn

from sonar_pipeline.cv import CONFIGS, fit_network
from sonar_pipeline.metrics import torch_cost
from sonar_pipeline.transformer import (
    TOKEN_BINS,
    GroupedQueryAttention,
    RMSNorm,
    SonarHybrid,
    SonarTransformer,
    apply_rope,
    rope_tables,
    swiglu_hidden,
)

REAL = {"lofar_frames": 11, "lofar_bins": 1001, "demon_bins": 301}


@pytest.mark.parametrize("cls, tokens", [(SonarTransformer, 52), (SonarHybrid, 64)])
@pytest.mark.parametrize("name", ["A", "B", "C", "D"])
def test_real_shapes_and_token_count(cls, tokens, name):
    config = CONFIGS["transformer"][name]
    keys = ("dim", "layers", "heads", "groups", "dropout")
    model = cls(**{k: config[k] for k in keys}, **REAL).eval()
    logits = model(torch.randn(2, 11, 1001), torch.randn(2, 301))
    assert logits.shape == (2, 4)
    assert model.tokens_per_segment == tokens
    cost = torch_cost(model, (torch.zeros(1, 11, 1001), torch.zeros(1, 301)))
    assert 0 < cost["parameters"] < 250_000  # every representative weight file < 1 MB


def test_transformer_tokens_are_40_hz_bands():
    model = SonarTransformer(**REAL)
    model.lofar_embed = nn.Identity()  # raw bands
    bins = torch.arange(1001.0).expand(1, 11, 1001)
    bands, _ = model.tokens(bins, torch.zeros(1, 301))
    assert bands.shape == (1, 50, 11 * TOKEN_BINS)
    for i in (0, 7, 49):
        expected = torch.arange(20.0 * i, 20.0 * i + 20).repeat(11)
        assert torch.equal(bands[0, i], expected)  # the 2000 Hz bin is never used


def test_hybrid_tokens_are_the_cnn_positions():
    model = SonarHybrid(**REAL)
    frequency, demon = model.tokens(torch.randn(2, 11, 1001), torch.randn(2, 301))
    assert frequency.shape == (2, 62, 32) and demon.shape == (2, 1, 32)


def test_rmsnorm_gives_unit_root_mean_square():
    x = torch.randn(3, 5, 16) * 7
    y = RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(dim=-1), torch.ones(3, 5), atol=1e-4)


def test_rope_score_depends_only_on_the_distance():
    head_dim = 8
    q, k = torch.randn(head_dim), torch.randn(head_dim)
    positions = torch.arange(20.0)
    cos, sin = rope_tables(positions, torch.ones(20), head_dim)

    def score(m, n):
        qm = apply_rope(q.expand(20, -1), cos, sin)[m]
        kn = apply_rope(k.expand(20, -1), cos, sin)[n]
        return float(qm @ kn)

    assert score(3, 1) == pytest.approx(score(13, 11), abs=1e-5)
    assert score(5, 5) == pytest.approx(float(q @ k), abs=1e-5)
    assert score(3, 1) != pytest.approx(score(3, 2), abs=1e-3)


def test_tokens_without_position_are_not_rotated():
    cos, sin = rope_tables(torch.tensor([0.0, 0.0, 4.0]), torch.tensor([0, 0, 1]), 4)
    x = torch.randn(3, 4)
    y = apply_rope(x, cos, sin)
    assert torch.equal(y[:2], x[:2]) and not torch.allclose(y[2], x[2])


def test_gqa_with_one_group_per_head_is_multi_head_attention():
    torch.manual_seed(0)
    dim, heads = 16, 4
    gqa = GroupedQueryAttention(dim, heads, heads, dropout=0.0).eval()
    mha = nn.MultiheadAttention(dim, heads, bias=False, batch_first=True).eval()
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([gqa.q.weight, gqa.k.weight, gqa.v.weight]))
        mha.out_proj.weight.copy_(gqa.out.weight)
    x = torch.randn(2, 7, dim)
    expected, _ = mha(x, x, x, need_weights=False)
    assert torch.allclose(gqa(x), expected, atol=1e-5)


def test_gqa_heads_of_a_group_share_keys_and_values():
    gqa = GroupedQueryAttention(16, heads=4, groups=2, dropout=0.0)
    assert gqa.k.out_features == 2 * 4  # 2 groups x head dimension 4
    kv = torch.randn(1, 2, 5, 4)
    shared = gqa.shared(kv)
    assert torch.equal(shared[:, 0], shared[:, 1]) and torch.equal(
        shared[:, 2], shared[:, 3]
    )
    assert torch.equal(shared[:, 0], kv[:, 0]) and torch.equal(shared[:, 2], kv[:, 1])


def test_swiglu_hidden_size_is_about_eight_thirds():
    assert swiglu_hidden(32) == 88 and swiglu_hidden(64) == 176


def test_attention_maps_are_kept_on_request():
    model = SonarTransformer(dim=32, layers=2, heads=2, groups=1, **REAL).eval()
    lofar, demon = torch.randn(3, 11, 1001), torch.randn(3, 301)
    model(lofar, demon)
    assert model.attention_maps() == [None, None]
    model.keep_attention()
    model(lofar, demon)
    maps = model.attention_maps()
    assert len(maps) == 2 and maps[0].shape == (3, 2, 52, 52)
    assert torch.allclose(maps[0].sum(dim=-1), torch.ones(3, 2, 52))


@pytest.mark.parametrize("model", ["transformer", "hybrid"])
def test_learns_the_toy(tmp_path, model):
    e = engine(tmp_path)
    data = e.net_data(e.partitions[0])
    config = {**CONFIGS[model]["B"], "lr": 3e-3, "dropout": 0.0}
    _, run = fit_network(model, config, data, 0, e.device, max_epochs=60, patience=60)
    assert run.best_val_f1 >= 0.9


@pytest.mark.parametrize("model", ["transformer", "hybrid"])
def test_same_seed_gives_the_same_run(tmp_path, model):
    e = engine(tmp_path)
    data = e.net_data(e.partitions[0])
    config = CONFIGS[model]["B"]
    _, a = fit_network(model, config, data, 3, e.device, max_epochs=2)
    _, b = fit_network(model, config, data, 3, e.device, max_epochs=2)
    assert [h["val_f1_macro"] for h in a.history] == [
        h["val_f1_macro"] for h in b.history
    ]
    for key in a.state:
        assert torch.equal(a.state[key], b.state[key])


@pytest.mark.parametrize("model", ["transformer_aug", "hybrid_aug", "cnn_aug"])
def test_same_seed_gives_the_same_run_on_cuda(tmp_path, model):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA here: run in the GPU image (compose.gpu.yaml)")
    e = engine(tmp_path, device_name="cuda")
    data = e.net_data(e.partitions[0])
    family = model.removesuffix("_aug")
    config = CONFIGS[family]["C" if family != "cnn" else "A"]
    augment = e.augment_for(model, e.partitions[0])
    _, a = fit_network(model, config, data, 3, e.device, augment, max_epochs=3)
    _, b = fit_network(model, config, data, 3, e.device, augment, max_epochs=3)
    assert [h["val_f1_macro"] for h in a.history] == [
        h["val_f1_macro"] for h in b.history
    ]
    for key in a.state:
        assert torch.equal(a.state[key], b.state[key])
