"""
Test that eta and _lambda_raw receive non-zero gradients via BPTT
through the Hebbian M-update chain.

Three cases:
  1. Non-recurrent (steps=1)  — eta/lambda should get ~zero grad
  2. Recurrent (steps=T, BPTT) — eta/lambda should get non-zero grad
  3. Recurrent, freeze_plasticity=True — eta/lambda should get zero grad (sanity)
"""

import sys
import torch
from tensordict import TensorDict
from torchrl.modules.tensordict_module.rnn import set_recurrent_mode

sys.path.insert(0, "/home/ewertj2/KAM/mpn-rl")
from mpn_module import MPNLayer


# ---------------------------------------------------------------------------
# Direct MPNLayer test (no TorchRL wrapper) — cleanest gradient check
# ---------------------------------------------------------------------------

def run_layer_test(T: int, freeze: bool, label: str):
    torch.manual_seed(0)
    layer = MPNLayer(input_dim=8, hidden_dim=16, freeze_plasticity=freeze)

    # Zero initial M (detached, as if from replay buffer)
    M = torch.zeros(2, 16, 8)

    inputs = [torch.randn(2, 8) for _ in range(T)]

    outputs = []
    for x in inputs:
        h, M = layer(x, M)
        outputs.append(h)

    loss = torch.stack(outputs, dim=1).sum()
    loss.backward()

    eta_grad   = layer.eta.grad
    lam_grad   = layer._lambda_raw.grad
    W_grad     = layer.W.grad

    print(f"\n[{label}]")
    print(f"  eta grad:     {eta_grad.item() if eta_grad is not None else None:.6f}"
          if eta_grad is not None else f"  eta grad:     None")
    print(f"  lambda grad:  {lam_grad.item() if lam_grad is not None else None:.6f}"
          if lam_grad is not None else f"  lambda grad:  None")
    print(f"  W grad norm:  {W_grad.norm().item():.6f}" if W_grad is not None else "  W grad norm:  None")

    if freeze:
        assert eta_grad is None or eta_grad.abs().item() < 1e-12, \
            "freeze_plasticity=True: eta should have zero/no grad"
        assert lam_grad is None or lam_grad.abs().item() < 1e-12, \
            "freeze_plasticity=True: lambda should have zero/no grad"
        print("  PASS: frozen plasticity correctly blocks eta/lambda grads")
    elif T == 1:
        # With a single step and detached M_0, eta/lambda get zero grad
        # (h_0 doesn't depend on eta; M_new is not in the loss)
        eta_val = eta_grad.abs().item() if eta_grad is not None else 0.0
        lam_val = lam_grad.abs().item() if lam_grad is not None else 0.0
        print(f"  NOTE: single-step — eta |grad|={eta_val:.2e}, lambda |grad|={lam_val:.2e}")
        print("  (expected ~0 for eta/lambda with detached M_0 and no future steps)")
    else:
        assert eta_grad is not None and eta_grad.abs().item() > 1e-10, \
            f"BPTT (T={T}): eta grad should be non-zero, got {eta_grad}"
        assert lam_grad is not None and lam_grad.abs().item() > 1e-10, \
            f"BPTT (T={T}): lambda grad should be non-zero, got {lam_grad}"
        print(f"  PASS: BPTT (T={T}) — eta and lambda receive non-zero gradients")


# ---------------------------------------------------------------------------
# MPNModule wrapper test — verifies set_recurrent_mode activates BPTT
# ---------------------------------------------------------------------------

def run_module_test():
    from mpn_torchrl_module import MPNModule

    torch.manual_seed(0)
    batch, T, input_size, hidden = 2, 10, 8, 16
    module = MPNModule(input_size=input_size, hidden_size=hidden, in_key="obs", out_key="embed")

    obs     = torch.randn(batch, T, input_size)
    state_m = torch.zeros(batch, T, hidden, input_size)   # detached, as from replay buffer
    is_init = torch.zeros(batch, T, 1, dtype=torch.bool)
    is_init[:, 0] = True

    td = TensorDict({
        "obs":             obs,
        "recurrent_state_m": state_m,
        "is_init":         is_init,
    }, batch_size=[batch, T])

    with set_recurrent_mode(True):
        out = module(td)

    loss = out["embed"].sum()
    loss.backward()

    eta_grad = module.mpn_layer.eta.grad
    lam_grad = module.mpn_layer._lambda_raw.grad

    print("\n[MPNModule wrapper — recurrent_mode=True]")
    print(f"  eta grad:    {eta_grad.item():.6f}" if eta_grad is not None else "  eta grad:    None")
    print(f"  lambda grad: {lam_grad.item():.6f}" if lam_grad is not None else "  lambda grad: None")

    assert eta_grad is not None and eta_grad.abs().item() > 1e-10, \
        f"MPNModule BPTT: eta grad should be non-zero, got {eta_grad}"
    assert lam_grad is not None and lam_grad.abs().item() > 1e-10, \
        f"MPNModule BPTT: lambda grad should be non-zero, got {lam_grad}"
    print("  PASS: MPNModule correctly activates BPTT via set_recurrent_mode")


if __name__ == "__main__":
    print("=" * 60)
    print("BPTT gradient flow test")
    print("=" * 60)

    run_layer_test(T=1,  freeze=False, label="MPNLayer  T=1  (non-recurrent, eta/lambda expected ~0)")
    run_layer_test(T=10, freeze=False, label="MPNLayer  T=10 (BPTT, eta/lambda expected non-zero)")
    run_layer_test(T=10, freeze=True,  label="MPNLayer  T=10 freeze_plasticity=True (expected 0)")

    run_module_test()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
