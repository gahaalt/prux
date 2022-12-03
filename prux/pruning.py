from flax.traverse_util import flatten_dict, unflatten_dict


class Pruning:
    @staticmethod
    def apply_exclusion(flat_masks, exclude_keywords):
        new_masks = {}
        for key, mask in flat_masks.items():
            str_key = str(key)
            for exclusion in exclude_keywords:
                if exclusion in str_key:
                    break
            else:
                new_masks[key] = mask
        return new_masks

    @classmethod
    def apply(cls, parameters, masks, exclude=()):
        flat_parameters = flatten_dict(parameters)
        flat_masks = flatten_dict(masks)

        if exclude:
            flat_masks = cls.apply_exclusion(flat_masks, exclude_keywords=exclude)

        for key in flat_masks:
            assert key in flat_parameters, f"{key} was not found in the model state!"

        pruned_parameters = {}
        for key, value in flat_parameters.items():
            if key in flat_masks:
                mask = flat_masks[key]
                value = value * mask
            pruned_parameters[key] = value

        return unflatten_dict(pruned_parameters)
