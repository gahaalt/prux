import jax.numpy as jnp
import jax.random
from flax.traverse_util import flatten_dict, unflatten_dict


class Criterion:
    @staticmethod
    def _concatenate_parameters(parameters):
        flat_parameters = flatten_dict(parameters)
        flat_parameters = [p.reshape(-1) for p in flat_parameters.values()]
        return jnp.concatenate(flat_parameters, axis=0)

    @staticmethod
    def _unconcatenate_mask(mask, parameters):
        flat_parameters = flatten_dict(parameters)
        chunk_sizes = jnp.array([p.size for p in flat_parameters.values()])
        chunk_sizes_cumsum = jnp.cumsum(chunk_sizes)
        chunk_shapes = [p.shape for p in flat_parameters.values()]

        masks = jnp.split(mask, indices_or_sections=chunk_sizes_cumsum)
        masks = [mask.reshape(shape) for mask, shape in zip(masks, chunk_shapes)]
        masks = {k: mask for k, mask in zip(flat_parameters, masks)}
        return masks

    @staticmethod
    def _get_mask_for_array(*args, **kwds):
        raise NotImplementedError("This is a base class used for inheritance!")

    @classmethod
    def get_masks(cls, parameters, *args, globally=True, **kwds):
        flat_parameters = flatten_dict(parameters)

        if globally:
            array = cls._concatenate_parameters(flat_parameters)
            mask = cls._get_mask_for_array(array, *args, **kwds)
            masks = cls._unconcatenate_mask(mask, parameters)
        else:
            masks = {}
            for parameter_name in flat_parameters:
                array = flat_parameters[parameter_name]
                masks[parameter_name] = cls._get_mask_for_array(array, *args, **kwds)
        return unflatten_dict(masks)


class MagnitudeCriterion(Criterion):
    @staticmethod
    def _get_mask_for_array(array, sparsity):
        flat_array = array.reshape(-1)
        to_prune = int(sparsity * flat_array.size)
        pruned_indices = jnp.argsort(jnp.abs(flat_array))[:to_prune]

        mask = jnp.ones(shape=flat_array.shape, dtype=array.dtype)
        mask = mask.at[pruned_indices].set(0)
        return mask.reshape(array.shape)


class RandomCriterion(Criterion):
    @staticmethod
    def _get_mask_for_array(array, sparsity, key):
        random_array = jax.random.uniform(key, shape=array.shape)
        return MagnitudeCriterion._get_mask_for_array(random_array, sparsity)


class SnipCriterion(Criterion):
    @staticmethod
    def _get_mask_for_array(array, grad, sparsity):
        snip_array = array * grad
        return MagnitudeCriterion._get_mask_for_array(snip_array, sparsity)
