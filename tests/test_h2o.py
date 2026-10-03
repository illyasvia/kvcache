import unittest
from types import SimpleNamespace

import torch

from compression.h2o import H2OConfig, H2OController


def make_cache(token_ids):
    keys = torch.tensor(token_ids, dtype=torch.float32).view(1, 1, -1, 1)
    values = (keys + 10).clone()
    return SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])


class H2OConfigTests(unittest.TestCase):
    def test_cache_size_is_fixed_budget_sum(self):
        config = H2OConfig(heavy_hitter_size=3, recent_size=2)
        self.assertEqual(config.cache_size, 5)

    def test_non_positive_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            H2OConfig(heavy_hitter_size=0, recent_size=1).validate()
        with self.assertRaises(ValueError):
            H2OConfig(heavy_hitter_size=1, recent_size=0).validate()


class H2OControllerTests(unittest.TestCase):
    def test_keeps_accumulated_heavy_hitters_and_recent_tokens(self):
        controller = H2OController(H2OConfig(heavy_hitter_size=2, recent_size=1))
        cache = make_cache([0, 1, 2, 3, 4])
        attention = torch.tensor(
            [[
                [[0.05, 0.50, 0.20, 0.15, 0.10]],
                [[0.05, 0.40, 0.30, 0.15, 0.10]],
            ]]
        )

        controller.update(0, cache, attention)

        self.assertEqual(cache.layers[0].keys.flatten().tolist(), [1.0, 2.0, 4.0])
        self.assertEqual(cache.layers[0].values.flatten().tolist(), [11.0, 12.0, 14.0])

        cache.layers[0].keys = torch.cat(
            [cache.layers[0].keys, torch.tensor([[[[5.0]]]])], dim=2
        )
        cache.layers[0].values = torch.cat(
            [cache.layers[0].values, torch.tensor([[[[15.0]]]])], dim=2
        )
        next_attention = torch.tensor(
            [[
                [[0.10, 0.10, 0.10, 0.70]],
                [[0.10, 0.10, 0.10, 0.70]],
            ]]
        )

        controller.update(0, cache, next_attention)

        self.assertEqual(cache.layers[0].keys.flatten().tolist(), [1.0, 2.0, 5.0])
        self.assertEqual(controller.stats["eviction_steps"], 2)
        self.assertEqual(controller.stats["layer_token_evictions"], 3)
        self.assertEqual(controller.stats["observed_layers"], 1)

    def test_does_not_evict_within_budget(self):
        controller = H2OController(H2OConfig(heavy_hitter_size=2, recent_size=1))
        cache = make_cache([0, 1, 2])
        attention = torch.full((1, 2, 1, 3), 1 / 3)

        controller.update(0, cache, attention)

        self.assertEqual(cache.layers[0].keys.shape[2], 3)
        self.assertEqual(controller.stats["eviction_steps"], 0)

    def test_reset_clears_scores_and_stats(self):
        controller = H2OController(H2OConfig(heavy_hitter_size=1, recent_size=1))
        cache = make_cache([0, 1, 2])
        controller.update(0, cache, torch.full((1, 1, 1, 3), 1 / 3))

        controller.reset()

        self.assertEqual(controller.stats["observed_layers"], 0)
        self.assertEqual(controller.stats["eviction_steps"], 0)


if __name__ == "__main__":
    unittest.main()
