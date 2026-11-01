"""GPU checks for step A0.4 (needs CUDA; the gdn/deltanet parts need flash-linear-attention).
Run: python -m tests.test_gpu
"""
import torch
import torch.nn.functional as F

from ww.arms import ARMS, check_params
from ww.model import LM
from ww.whatwhen import ww_chunked, ww_sequential

DEV = torch.device("cuda")


def rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def test_cuda():
    assert torch.cuda.is_available(), "no CUDA device"
    print("  device:", torch.cuda.get_device_name(), "| torch", torch.__version__)


def test_ww_core_on_gpu_matches_fp64_definition():
    from tests.test_whatwhen import random_inputs
    for decay in (False, True):
        k, q, v, la, _ = random_inputs(B=2, H=4, T=1024, V=1, decay=decay)
        y_ref, _ = ww_sequential(k, q, v, la)
        leaves = [t.float().to(DEV).requires_grad_(True) for t in (k, q, v)]
        y, _ = ww_chunked(*leaves, None if la is None else la.float().to(DEV), chunk=32)
        assert rel(y.double().cpu(), y_ref) < 1e-5, rel(y.double().cpu(), y_ref)
        y.sum().backward()
        assert all(torch.isfinite(t.grad).all() for t in leaves)


def test_fla_arms_run_and_are_causal():
    for arm in ("gdn", "deltanet"):
        torch.manual_seed(0)
        model = LM(ARMS[arm]["model"]).to(DEV)
        for m in model.modules():  # non-zero output projections so the causality check means something
            if getattr(m, "is_residual_out", False):
                torch.nn.init.normal_(m.weight, std=0.02)
        x = torch.randint(0, 128, (2, 1024), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 128), x.reshape(-1))
        loss.backward()
        assert torch.isfinite(loss)
        x2 = x.clone()
        x2[:, 600:] = torch.randint(0, 128, (2, 424), device=DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            a, _ = model(x)
            b, _ = model(x2)
        diff = (a[:, :600].float() - b[:, :600].float()).abs().max().item()
        assert diff < 1e-2, f"{arm}: prefix logits changed by {diff} when only the future changed"
        print(f"  {arm}: ok (prefix change {diff:.1e})")


def test_param_match_including_fla():
    bad = check_params(device="cuda")
    assert not bad, f"outside ±3% non-embedding params: {bad} -> adjust ffn_hidden in ww/arms.py"


def test_one_bf16_update_per_main_arm_at_full_size():
    for arm in ("tpp", "gdn", "ww"):
        torch.manual_seed(0)
        model = LM(ARMS[arm]["model"]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        x = torch.randint(0, 128, (8, 1025), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, 128), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), arm
        print(f"  {arm}: initial loss {loss.item() / 0.6931:.3f} bits (expect ~7.0)")
        del model, opt
        torch.cuda.empty_cache()


def test_bf16_decode_close_to_full_forward():
    for arm in ("tpp", "ww"):
        torch.manual_seed(0)
        model = LM(ARMS[arm]["model"]).to(DEV).eval()
        for m in model.modules():
            if getattr(m, "is_residual_out", False):
                torch.nn.init.normal_(m.weight, std=0.02)
        x = torch.randint(0, 128, (2, 300), device=DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            full, _ = model(x)
            logits, st = model(x[:, :200], want_state=True)
            outs = [logits]
            for t in range(200, 300):
                lt, st = model(x[:, t:t + 1], states=st, want_state=True)
                outs.append(lt)
        diff = (torch.cat(outs, 1).float() - full.float()).abs().max().item()
        assert diff < 0.1, f"{arm}: decode differs from full forward by {diff}"
        print(f"  {arm}: decode vs full forward max |dlogit| = {diff:.1e} (bf16)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
