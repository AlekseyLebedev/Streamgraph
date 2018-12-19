import unittest
from streamgraph import StreamGraph
from streamgraph import streamgraph as sg_internal
import numpy as np


class TestSilhouette(unittest.TestCase):
    def test_avg(self):
        inp = np.arange(12).reshape((4, 3))
        res = sg_internal._avg_silhouette(inp)
        res = list(res)
        self.assertSequenceEqual(res, [-1.5, -6, -10.5, -15])

    def test_wiggle(self):
        inp = np.arange(12).reshape((4, 3))
        res = sg_internal._wiggle_silhouette(inp)
        expected = []
        expected.append(3 * 0 + 2 * 1 + 1 * 2)
        expected.append(3 * 3 + 2 * 4 + 1 * 5)
        expected.append(3 * 6 + 2 * 7 + 1 * 8)
        expected.append(3 * 9 + 2 * 10 + 1 * 11)
        expected = -np.array(expected) / 4
        expected = list(expected)
        res = list(res)
        self.assertSequenceEqual(res, expected)

    def test_weighted_wiggle(self):
        inp = np.array([[1, 1, 1], [1, 2, 3], [3, 2, 1]]).transpose()

        # Производные
        # [0, 0, 0]
        # [1, 1, 1]
        # [-1, -1, -1]

        # Сумма
        #  [0, 0, 0]
        #  [0.5, 1, 1.5]
        #  [1.5, 1, 0.5]
        #   итог [-2.5, -2, -2.5]
        #  нормируем / 5
        # [-0,4, -0,4, -0.4]

        expected = [-(3 * 1 + 2 * 1 + 1 * 3) / 4]
        expected.append(expected[-1] - 0.4)
        expected.append(expected[-1] - 0.4)
        expected = list(expected)
        res = sg_internal._weighted_wiggle_silhouette(inp)
        res = list(res)
        self.assertSequenceEqual(res, expected)

    def test_weighted_wiggle_special(self):
        # Граничный случай
        inp = np.arange(8).reshape((2, 4))
        res1 = sg_internal._weighted_wiggle_silhouette(inp)
        res2 = sg_internal._avg_silhouette(inp)
        self.assertSequenceEqual(list(res1), list(res2))
        inp = np.arange(8).reshape((1, 8))
        res1 = sg_internal._weighted_wiggle_silhouette(inp)
        res2 = sg_internal._avg_silhouette(inp)
        self.assertSequenceEqual(list(res1), list(res2))


class TestStreamGraph(unittest.TestCase):
    def test_silhouette_abracadabra_fails(self):
        with self.assertRaises(AssertionError):
            StreamGraph('abracadabra')

    def test_silhouette_ok(self):
        StreamGraph('avg')
        StreamGraph('wiggle')
        StreamGraph('weighted_wiggle')

    def test_draw_im(self):
        sg = StreamGraph(ordering=None)
        x = np.transpose([[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        res, _ = sg.draw_im(x, colors=colors, im_size=(3, 4))
        ref = [[[1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.]],

               [[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.],
                [0., 0., 1.]],

               [[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.],
                [0., 0., 1.], ]]
        self.assertSequenceEqual(list(res.reshape(-1)), list(np.reshape(ref, -1)))

    def test_draw_im_dummy(self):
        sg = StreamGraph()
        x = np.arange(56).reshape(7, 8)
        res, _ = sg.draw_im(x)
        self.assertEquals(res.shape, (512, 1024, 3))


if __name__ == '__main__':
    unittest.main()
