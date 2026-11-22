import unittest

from finaudit.rng import Rng


class RngTests(unittest.TestCase):
    def test_same_seed_same_stream(self):
        a = Rng(42)
        b = Rng(42)
        self.assertEqual([a.random() for _ in range(20)], [b.random() for _ in range(20)])
        self.assertEqual([a.randint(0, 1000) for _ in range(20)], [b.randint(0, 1000) for _ in range(20)])

    def test_different_seed_differs(self):
        a = Rng(1)
        b = Rng(2)
        self.assertNotEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_fork_is_independent(self):
        root = Rng(7)
        left_a = root.fork("left")
        left_b = root.fork("left")
        right = root.fork("right")
        left_draws = [left_a.random() for _ in range(5)]
        self.assertEqual([left_b.random() for _ in range(5)], left_draws)
        right_draws = [right.random() for _ in range(5)]
        self.assertNotEqual(left_draws, right_draws)
        self.assertEqual(root.fork("left").random(), left_draws[0])

    def test_shuffle_and_sample_are_deterministic(self):
        a = list(range(30))
        b = list(range(30))
        Rng(3).shuffle(a)
        Rng(3).shuffle(b)
        self.assertEqual(a, b)
        self.assertEqual(Rng(9).sample(list(range(50)), 7), Rng(9).sample(list(range(50)), 7))
        self.assertEqual(len(set(Rng(9).sample(list(range(50)), 7))), 7)

    def test_choice_and_weighted(self):
        rng = Rng(5)
        self.assertIn(rng.choice(["a", "b", "c"]), {"a", "b", "c"})
        with self.assertRaises(ValueError):
            rng.choice([])
        self.assertIn(rng.weighted(["a", "b"], [1, 1]), {"a", "b"})
        with self.assertRaises(ValueError):
            rng.weighted([], [])

    def test_uniform_bounds(self):
        rng = Rng(11)
        for _ in range(100):
            value = rng.uniform(-2.5, 4.5)
            self.assertGreaterEqual(value, -2.5)
            self.assertLessEqual(value, 4.5)

    def test_hex_length(self):
        self.assertEqual(len(Rng(1).hex(3)), 6)


if __name__ == "__main__":
    unittest.main()
