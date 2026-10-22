"""Portable reduced operators and versioned HDF5 storage."""

from dataclasses import asdict, dataclass
import hashlib
import json

import h5py
import numpy as np

from . import __version__
from .problem import Component


@dataclass
class Compression:
    """Mean plus coefficient-weighted modes, with grid axes leading coefficients."""

    mean: np.ndarray
    modes: np.ndarray
    coeff_grid: np.ndarray
    singular_values: np.ndarray


@dataclass
class ComponentData:
    matrix: Compression
    source: Compression


@dataclass
class BlockData:
    M0: np.ndarray
    s0: np.ndarray
    LX: np.ndarray
    components: list[ComponentData]


@dataclass
class EmulatorData:
    param_names: list[str]
    components: list[Component]
    blocks: list[BlockData]
    projection: str = 'hermitian'


def components_hash(components):
    """Fingerprint ordered definitions, including reference strengths and grids."""
    return hashlib.sha256(repr(list(components)).encode('utf-8')).hexdigest()


def save(data: EmulatorData, path):
    """Write schema 1. Observables are code and are supplied separately on load."""
    with h5py.File(path, 'w') as file:
        file.attrs['schema'] = 1
        file.attrs['dream_version'] = __version__
        file.attrs['components_hash'] = components_hash(data.components)
        file.attrs['components'] = json.dumps([asdict(c) for c in data.components])
        file.attrs['param_names'] = json.dumps(data.param_names)
        file.attrs['projection'] = data.projection
        blocks = file.create_group('blocks')
        for index, block in enumerate(data.blocks):
            group = blocks.create_group(str(index))
            for name in ('M0', 's0', 'LX'):
                group.create_dataset(name, data=getattr(block, name))
            terms = group.create_group('components')
            for f, component in enumerate(block.components):
                term = terms.create_group(str(f))
                for name in ('matrix', 'source'):
                    compressed = term.create_group(name)
                    for field in ('mean', 'modes', 'coeff_grid', 'singular_values'):
                        compressed.create_dataset(field, data=getattr(getattr(component, name), field))


def load(path, *, components=None) -> EmulatorData:
    """Read a database and check its definition hash, optionally against an adapter.

    The hash detects mismatched component definitions, not changed adapter code.
    """
    with h5py.File(path, 'r') as file:
        if file.attrs.get('schema') != 1:
            raise ValueError(f"unsupported database schema: {file.attrs.get('schema')}")
        definitions = []
        for value in json.loads(file.attrs['components']):
            value['geometry'] = tuple(value['geometry'])
            value['grid'] = tuple(tuple(axis) for axis in value['grid'])
            definitions.append(Component(**value))
        expected = file.attrs.get('components_hash')
        if components_hash(definitions) != expected:
            raise ValueError('components_hash mismatch in stored definitions')
        if components is not None and components_hash(components) != expected:
            raise ValueError('components_hash mismatch with supplied adapter definitions')
        blocks = []
        for index in range(len(file['blocks'])):
            group = file['blocks'][str(index)]
            terms = []
            for f in range(len(definitions)):
                term = group['components'][str(f)]
                compressed = []
                for name in ('matrix', 'source'):
                    compressed.append(Compression(**{
                        field: term[name][field][()]
                        for field in ('mean', 'modes', 'coeff_grid', 'singular_values')
                    }))
                terms.append(ComponentData(*compressed))
            blocks.append(BlockData(*(group[name][()] for name in ('M0', 's0', 'LX')), terms))
        return EmulatorData(json.loads(file.attrs['param_names']), definitions, blocks,
                            str(file.attrs['projection']))
