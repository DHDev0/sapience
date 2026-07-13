"""The spiking substrate: LIFCell / ALIFCell — sequence run == stepping, growth preserves function."""
import torch
from brain.spiking import LIFCell, ALIFCell, spike

DEV = torch.device("cpu")


def test_surrogate_spike_forward_and_grad():
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    s = spike(x)
    assert torch.equal(s, torch.tensor([0.0, 1.0, 1.0]))          # Heaviside at 0
    s.sum().backward()
    assert (x.grad > 0).all()                                     # fast-sigmoid surrogate is positive


def test_run_seq_equals_stepping():
    for Cell in (LIFCell, ALIFCell):
        torch.manual_seed(1)
        c = Cell(16, 24)
        x = torch.randn(4, 12, 16); st = c.init_state(4, DEV)
        stepped, s2 = [], st
        for t in range(12):
            out, s2 = c(x[:, t], s2); stepped.append(out)
        stepped = torch.stack(stepped, 1)
        seq, _, _ = c.run_seq(x, st)
        assert torch.allclose(stepped, seq, atol=1e-6), f"{Cell.__name__} run_seq != stepped"


def test_grow_is_identity_preserving():
    for Cell in (LIFCell, ALIFCell):
        torch.manual_seed(2)
        c = Cell(16, 24)
        x = torch.randn(3, 8, 16); st = c.init_state(3, DEV)
        before, _, _ = c.run_seq(x, st)
        c.grow(16)                                                # new neurons have ~0 output
        st2 = c.init_state(3, DEV)
        after, _, _ = c.run_seq(x, st2)
        # existing neurons' spikes unchanged (new ones appended as extra columns)
        assert torch.allclose(before, after[..., :before.shape[-1]], atol=1e-6)
        assert c.hid == 40
