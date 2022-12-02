import jax.numpy as jnp
import jax.random
from flax.traverse_util import flatten_dict, unflatten_dict


class Criterion:
    def _concatenate_parameters(self, parameters):
        flat_parameters = flatten_dict(parameters)
        flat_parameters = [p.reshape(-1) for p in flat_parameters.values()]
        return jnp.concatenate(flat_parameters, axis=0)

    def _unconcatenate_mask(self, mask, parameters):
        flat_parameters = flatten_dict(parameters)
        chunk_sizes = jnp.array([p.size for p in flat_parameters.values()])
        chunk_sizes_cumsum = jnp.cumsum(chunk_sizes)
        chunk_shapes = [p.shape for p in flat_parameters.values()]

        masks = jnp.split(mask, indices_or_sections=chunk_sizes_cumsum)
        masks = [mask.reshape(shape) for mask, shape in zip(masks, chunk_shapes)]
        masks = {k: mask for k, mask in zip(flat_parameters, masks)}
        return masks

    def _get_mask_for_array(self, *args, **kwds):
        raise NotImplementedError

    def get_masks(self, parameters, *args, globally=True, **kwds):
        flat_parameters = flatten_dict(parameters)

        if globally:
            array = self._concatenate_parameters(flat_parameters)
            mask = self._get_mask_for_array(array, *args, **kwds)
            masks = self._unconcatenate_mask(mask, parameters)
        else:
            masks = {}
            for parameter_name in flat_parameters:
                array = flat_parameters[parameter_name]
                masks[parameter_name] = self._get_mask_for_array(array, *args, **kwds)
        return unflatten_dict(masks)


class MagnitudeCriterion(Criterion):
    def _get_mask_for_array(self, array, sparsity):
        flat_array = array.reshape(-1)
        to_prune = int(sparsity * flat_array.size)
        pruned_indices = jnp.argsort(flat_array)[:to_prune]

        mask = jnp.ones(shape=array.shape, dtype=array.dtype)
        mask = mask.at[pruned_indices].set(0)
        return mask


class RandomCriterion(Criterion):
    def _get_mask_for_array(self, array, sparsity, key):
        fake_array = jax.random.uniform(key, shape=array.shape)
        return MagnitudeCriterion()._get_mask_for_array(fake_array, sparsity)
