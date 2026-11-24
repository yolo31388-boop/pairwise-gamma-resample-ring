import pytest
import imaging.gamma_resample_ring as mod

CASES = [
    ("linear_space", (1,)),
    ("truncate_kernel", (1,)),
    ("uniform_factor", (1,)),
    ("fixed_order", (1,)),
    ("premult_alpha", (1,)),
    ("load", (1,)),
    ("resize", (1,)),
    ("crop", (1,)),
    ("export", (1,)),
    ("kernel_width", (1,)),
    ("validate_scale", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
