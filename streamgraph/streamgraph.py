import numpy as np
import pandas as pd
import colorsys
from scipy import interpolate

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt


def _avg_silhouette(x):
    """
    Создает симметричный граф
    """
    return -x.sum(axis=1) / 2


def _wiggle_silhouette(x):
    """
    Минимизирует дисперсию производных
    """
    n = x.shape[1]
    return -(np.arange(n, 0, -1) * x).sum(axis=1) / (n + 1)


def _weighted_wiggle_silhouette(x):
    """
    Минимизирует взвешенную дисперсию производных (веса по толщине слоев)
    """

    if x.shape[0] < 3:
        # проблемы с интерполяцией производных
        return _avg_silhouette(x)

    # Считаем аппроксимацию производных
    diffs = np.zeros(x.shape)
    diffs[1:-1] = (x[2:] - x[:-2]) / 2
    diffs[0] = (-3 * x[0] + 4 * x[1] - x[2]) / 2
    diffs[-1] = (x[-3] - 4 * x[-2] + 3 * x[-1]) / 2

    norm = x.sum(axis=1)
    norm = -1 / norm

    # Производная нижнего края
    g_diff = np.zeros((x.shape[0]))
    for i in range(x.shape[1]):
        g_diff += (0.5 * diffs[:, i] + diffs[:, :i].sum(axis=1)) * x[:, i]
    g_diff *= norm

    # Численно интегрируем
    g = [_wiggle_silhouette(x[:1])[0]]
    for i in range(1, x.shape[0]):
        g.append(g[-1] + (g_diff[i - 1] + g_diff[i]) * 0.5)
    return np.array(g)


_silhouette = {
    'avg': _avg_silhouette,
    'wiggle': _wiggle_silhouette,
    'weighted_wiggle': _weighted_wiggle_silhouette,
}


def _inside_out_ordering(x):
    weighted = np.array(x)
    for i in range(x.shape[1]):
        weighted[:, i] *= i
    weighted = weighted.sum(axis=0)
    order = list(np.argsort(weighted))
    result = []
    for i in order:
        result.append(i)
        result = result[::-1]
    return result


def _identity_ordering(x):
    return range(x.shape[1])


_ordering = {
    'inside_out': _inside_out_ordering
}


class StreamGraph:
    def __init__(self, silhouette='weighted_wiggle', ordering='inside_out'):
        assert silhouette in _silhouette
        self._silhouette = _silhouette[silhouette]
        assert ordering is None or ordering in _ordering
        if ordering is None:
            self._order_func = _identity_ordering
        else:
            self._order_func = _ordering[ordering]

    def draw(self, x, colors=None, texts=None):
        if isinstance(x, pd.DataFrame):
            assert texts is None
            texts = [str(t) for t in x.columns]
        im, colors = self.draw_im(x, colors=colors)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey='none', sharex='none')
        plt.xticks([])
        plt.yticks([])
        ax1.imshow(im)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_adjustable('box')
        plt.setp(ax1.spines.values(), color=(1, 1, 1))
        plt.setp(ax2.spines.values(), color=(1, 1, 1))
        if texts is not None:
            patches = [mpatches.Patch(color=c, label=texts[i]) for i, c in enumerate(colors)]
            legend = ax2.legend(handles=patches)
            ax2.set_adjustable('box')
            ax2.set_aspect(5)
        plt.show()

    def draw_im(self, x, im_size=None, colors=None):
        x = self._prepare_data(x)
        x, permutation = self._reorder(x)
        if im_size is None:
            im_size = (512, 1024)
        if colors is None:
            colors = self.create_colors(x)
        else:
            colors = [colors[i] for i in permutation]
        intervals = self._intervals(x)
        intervals = self._interpolate(intervals, im_size[1])
        return self._fill_image(im_size, intervals, colors), self._back_order(permutation, colors)

    def create_colors(self, x):
        x = self._prepare_data(x)
        hue_range = (30 / 360, 300 / 360)
        sums = x.sum(axis=0)
        sums /= np.max(sums)
        hsv = [(self._value_to_range(i / len(sums), hue_range),
                self._value_to_range(v, (0.5, 1)),
                0.75) for i, v in enumerate(sums)]
        rgb = [colorsys.hsv_to_rgb(*v) for v in hsv]
        return rgb

    def _back_order(self, permutation, values):
        original_order = [0 for _ in permutation]
        for i, pos in enumerate(permutation):
            original_order[pos] = i
        return [values[i] for i in original_order]

    def _reorder(self, x):
        permutation = self._order_func(x)
        x = np.transpose([x[:, i] for i in permutation])
        return x, permutation

    def _intervals(self, x):
        result = np.zeros((x.shape[1] + 1, x.shape[0]))
        result[0] = self._silhouette(x)
        for i in range(x.shape[1]):
            result[i + 1] = result[i] + x[:, i]
        result -= result.min()
        result /= result.max()
        return result

    def _interpolate(self, intervals, width):
        result = np.zeros((intervals.shape[0], width))
        target_x = np.linspace(0, intervals.shape[1], width)
        source_x = np.arange(intervals.shape[1])
        for i in range(intervals.shape[0]):
            spline = interpolate.splrep(source_x, intervals[i])
            result[i] = interpolate.splev(target_x, spline)
        return result

    def _fill_image(self, im_size, intervals, colors):
        assert intervals.shape[1] == im_size[1]
        assert len(colors) + 1 == intervals.shape[0]
        intervals -= intervals.min()
        intervals /= intervals.max()
        intervals *= im_size[0]
        im = np.ones((*im_size, 3))
        for x in range(intervals.shape[1]):
            for i, (start, end) in enumerate(zip(intervals[:-1, x], intervals[1:, x])):
                start = int(start)
                end = int(end)
                start = max(0, start)
                end = min(im.shape[0], end)
                im[start:end, x] = colors[i]
        return im

    def _value_to_range(self, value, range):
        start, stop = range
        return start + value * (stop - start)

    def _prepare_data(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        x = np.array(x, np.float)
        assert len(x.shape) == 2
        return x
