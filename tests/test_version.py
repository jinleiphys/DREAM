import jax

import dream


def test_version():
    assert dream.__version__ == "0.1.0"


def test_x64_enabled():
    assert jax.config.jax_enable_x64 is True


def test_readme_quickstart():
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    blocks = re.findall(r'```python\n(.*?)```', readme, re.S)
    assert blocks, 'README needs an executable Python quick start'
    namespace = {}
    for block in blocks:
        exec(block, namespace)
    assert namespace['check'].passed
