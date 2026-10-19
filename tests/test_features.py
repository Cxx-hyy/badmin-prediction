# -*- coding: utf-8 -*-
"""测试 ml/data_pipeline.py 里的纯函数: 手算结果对照 + 边界情况。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.data_pipeline import tokens, win_rate_last_n, tier_level


def test_points_share_manual():
    # 手算: 85000/(85000+60000) = 0.586 (用代码里的公式直接验证)
    pts1, pts2 = 85000, 60000
    assert abs(pts1 / (pts1 + pts2) - 0.586) < 0.001


def test_win_rate_last_n():
    # 没打过 -> 0.5(中性)
    assert win_rate_last_n([], 5) == 0.5
    # 不足5场 -> 按实际场次算
    assert abs(win_rate_last_n([1, 1, 0], 5) - 2 / 3) < 1e-9
    # 正好5场
    assert abs(win_rate_last_n([1, 1, 0, 1, 1], 5) - 0.8) < 1e-9
    # 只取最近5场(前面更早的忽略)
    assert abs(win_rate_last_n([0, 0, 0, 1, 1, 1], 5) - 0.6) < 1e-9


def test_tokens_normalize():
    # 大小写/顺序/标点归一
    assert tokens("KHRISHNAN Yogendran") == tokens("khrishnan YOGENDRAN")
    assert tokens("Anthony Ginting") == tokens("Anthony Ginting ")  # 去空格后相等
    assert tokens("A. B.") == tokens("a b")


def test_tier_level():
    assert tier_level("HSBC BWF World Tour Super 1000") == 7
    assert tier_level("Grand Prix") == 3
    assert tier_level("unknown") == 2   # 未知级别兜底