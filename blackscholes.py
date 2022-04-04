#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:52:18 2021

@author: kuanhungwang
"""
from itertools import product

import numpy as np
import pandas as pd
from numpy import exp
from numpy.lib.scimath import log, sqrt
# 定義ＢＳ公式
from scipy import stats

# import black76

cdf = stats.norm.cdf
pdf = stats.norm.pdf


def dcf(r, t):
    return exp(-r * t)


def implied_interest(s, f, t, q):
    """
    從期貨與現貨推算隱含利率
    """
    return q + log(f / s) / t


def implied_dividend(s, f, t, r):
    '''
    從期貨與現貨推算隱含連續股利殖利率
    '''
    return r - log(f / s) / t


def implied_future(s, t, r, q):
    """
    從利率、連續股利殖利率推算期貨
    """
    return s * exp((r - q) * t)


def implied_spot(f, t, r, q):
    """
    從利率、連續股利殖利率推算現貨
    """
    return f * exp((q - r) * t)


def d1(s, k, t, r, q, v):
    return (log(s / k) + (r - q + 0.5 * v ** 2) * t) / (v * sqrt(t))


def d2(s, k, t, r, q, v):
    return d1(s, k, t, r, q, v) - v * sqrt(t)


def price(s, k, t, r, q, v, isCall):
    isPut = np.bitwise_not(isCall)
    return call(s, k, t, r, q, v) * isCall + put(s, k, t, r, q, v) * isPut


def call(s, k, t, r, q, v):
    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    return s * exp(-q * t) * cdf(d1_) - k * exp(-r * t) * cdf(d2_)


def put(s, k, t, r, q, v):
    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    return k * exp(-r * t) * cdf(-d2_) - s * exp(-q * t) * cdf(-d1_)


# greeks
# first order


def delta(s, k, t, r, q, v, isCall):
    """
    應變：price, 變數：s
    """
    isPut = np.bitwise_not(isCall)
    d1_ = d1(s, k, t, r, q, v)
    return exp(-q * t) * cdf(d1_) * isCall + -exp(-q * t) * cdf(-d1_) * isPut


def vega(s, k, t, r, q, v, isCall):
    """
    應變：price, 變數：v
    """
    d2_ = d2(s, k, t, r, q, v)
    return k * exp(-r * t) * pdf(d2_) * sqrt(t)


def theta(s, k, t, r, q, v, isCall):
    '''
    應變：price, 變數：t
    '''
    isPut = np.bitwise_not(isCall)
    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    A = -s * pdf(d1_) * v / (2 * sqrt(t))
    B = -r * k * exp(-r * t) * cdf(d2_) * isCall + \
        r * k * exp(-r * t) * cdf(-d2_) * isPut
    C = q * s * exp(-q * t) * cdf(d1_) * isCall + \
        -q * s * exp(-q * t) * cdf(-d1_) * isPut
    return A + B + C


def rho(s, k, t, r, q, v, isCall):
    '''
    應變：price, 變數：r
    '''
    d2_ = d2(s, k, t, r, q, v)

    sign = isCall * 2 - 1
    return sign * k * t * exp(-r * t) * cdf(sign * d2_)


# second order


def gamma(s, k, t, r, q, v, isCall):
    '''
    應變：delta, 變數：s (導數分母：s*s)
    '''
    d1_ = d1(s, k, t, r, q, v)
    return exp(-q * t) * pdf(d1_) / (s * v * sqrt(t))


def vanna(s, k, t, r, q, v, isCall):
    '''
    應變：delta, 變數：v，或
    應變：vega, 變數：s
    (導數分母：s*v)
    '''
    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    return -exp(-q * t) * pdf(d1_) * d2_ / v


def charm(s, k, t, r, q, v, isCall):
    """
    應變：delta, 變數：t，或
    應變：theta, 變數：s
    (導數分母：s*t)
    """
    sign = isCall * 2 - 1
    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    A = (2 * (r - q) * t - d2_ * v * sqrt(t)) / (2 * t * v * sqrt(t))
    return sign * q * exp(-q * t) * cdf(sign * d1_) - exp(-q * t) * pdf(d1_) * A


def vomma(s, k, t, r, q, v, isCall):
    """
    應變：vega, 變數：v
    (導數分母：v*v)
    """
    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    return s * exp(-q * t) * pdf(d1_) * sqrt(t) * d1_ * d2_ / v


def veta(s, k, t, r, q, v, isCall):
    '''
    應變：vega, 變數：t
    (導數分母：v*t)
    '''
    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    A = -s * exp(-q * t) * pdf(d1_) * sqrt(t)
    B = (r - q) * d1_ / (v * sqrt(t))
    C = (1 + d1_ * d2_) / (2 * t)
    return A * (q + B - C)


# third-order greeks


def speed2(s, k, t, r, q, v, isCall):
    '''
    選擇權價值對spot的三次微分
    '''
    d1_ = d1(s, k, t, r, q, v)
    A = s * exp(-q * t) * pdf(d1_) / (s ** 2 * v * sqrt(t))
    B = (d1_ / (v * sqrt(t)) + 1)
    return A * B


def speed(s, k, t, r, q, v, isCall):
    '''
    應變：gamma, 變數：s
    (導數分母：s*s*s)
    '''
    d1_ = d1(s, k, t, r, q, v)
    gamma_ = gamma(s, k, t, r, q, v, isCall)
    return - gamma_ / s * (d1_ / (v * sqrt(t)) + 1)


def zomma(s, k, t, r, q, v, isCall):
    '''
    應變：gamma, 變數：v
    (導數分母：s*s*v)
    '''

    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    gamma_ = gamma(s, k, t, r, q, v, isCall)
    return gamma_ * (d1_ * d2_ - 1) / v


def color(s, k, t, r, q, v, isCall):
    """
    應變：gamma, 變數：t
    (導數分母：s*s*t)
    """

    d1_ = d1(s, k, t, r, q, v)
    d2_ = d2(s, k, t, r, q, v)
    A = -exp(-q * t) * pdf(d1_) / (2 * s * t * v * sqrt(t))
    B = (2 * (r - q) * t - d2_ * v * sqrt(t)) * d1_ / (v * sqrt(t))
    return A * (2 * q * t + 1 + B)


def get_dummy_option_df(s, K, T, r, v):
    calls = pd.DataFrame(columns=("s", "k", "t", "r", "v", "price"))
    puts = calls.copy()
    i = 0
    for k, t in product(K, T):
        price = call(s, k, t, r, 0, v)
        df = pd.DataFrame(
            dict(s=s, k=k, t=t, r=r, v=v, price=price), index=[i])
        calls = calls.append(df)
        i = i + 1

        price = put(s, k, t, r, 0, v)
        df = pd.DataFrame(
            dict(s=s, k=k, t=t, r=r, v=v, price=price), index=[i])
        puts = puts.append(df)
        i = i + 1
    return calls, puts


if __name__ == "__main__":
    S = 10500
    K = np.arange(10000, 11000, 200)
    T = np.linspace(0.1, 0.5, 4)
    interest_rate = 0.001
    sigma = 0.3

    f, s, k, t, r, q, v = 15600, 15000, 15000, 0.08, 0.01, 0.025, 0.3
    input1 = (f, k, t, r, v)
    input2 = (implied_spot(f, t, r, q), k, t, r, q, v)
    input3 = (s, k, t, r, implied_dividend(s, f, t, r), v)
    input4 = (s, k, t, implied_interest(s, f, t, q), q, v)
    print(black76.call(*input1))
    print(call(*input4))
