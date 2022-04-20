import pytest
import numpy as np
from importlib import reload


import facet

reload(facet)


def test_get_strike_dip_from_normal_1():
    n = np.array([0.0, 0.0, 0.5])

    strike, dip = facet.source_builder.get_strike_dip_from_normal(n)

    np.testing.assert_almost_equal(strike, 0.0)
    np.testing.assert_almost_equal(dip, 60.0)


test_get_strike_dip_from_normal_1()


def test_circular_strike_dip_normal():

    strikes = (0.0, 7.0, 60.0, 90.0, 137.0, 180.0, 196.0, 270.0, 299.0)
    dips = (2.0, 35.0, 45.0, 60.0, 85.0)

    for strike in strikes:
        for dip in dips:
            normal = facet.source_builder.get_normal_from_strike_dip(
                strike, dip
            )
            strike_, dip_ = facet.source_builder.get_strike_dip_from_normal(
                normal
            )

            np.testing.assert_almost_equal(strike, strike_)
            np.testing.assert_almost_equal(dip, dip_)


test_circular_strike_dip_normal()


def test_get_rake_from_shear_components_1():
    strike_shear = 0.0
    dip_shear = 1.0

    rake = facet.source_builder.get_rake_from_shear_components(
        strike_shear, dip_shear
    )

    np.testing.assert_almost_equal(rake, 90.0)


test_get_rake_from_shear_components_1()


def test_get_optimal_fault_plane_extension_1():
    T = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1.0]])

    strike_30, dip_30, rake_30 = facet.source_builder.get_optimal_fault_plane(
        T, 30.0
    )

    np.testing.assert_almost_equal(0.0, strike_30)
    np.testing.assert_almost_equal(60.0, dip_30)
    np.testing.assert_almost_equal(-90.0, rake_30)

    strike_10, dip_10, rake_10 = facet.source_builder.get_optimal_fault_plane(
        T, 10.0
    )
    np.testing.assert_almost_equal(0.0, strike_10)
    np.testing.assert_almost_equal(50.0, dip_10)
    np.testing.assert_almost_equal(-90.0, rake_10)


def test_get_optimal_fault_plane_extension_2():
    T = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1.0]])

    strike_30, dip_30, rake_30 = facet.source_builder.get_optimal_fault_plane(
        T, 30.0
    )

    np.testing.assert_almost_equal(270.0, strike_30)
    np.testing.assert_almost_equal(60.0, dip_30)
    np.testing.assert_almost_equal(-90.0, rake_30)


test_get_optimal_fault_plane_extension_1()
test_get_optimal_fault_plane_extension_2()


def test_get_optimal_fault_plane_contraction_1():
    T = np.array([[-1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])

    strike_30, dip_30, rake_30 = facet.source_builder.get_optimal_fault_plane(
        T, 30.0
    )

    np.testing.assert_almost_equal(0.0, strike_30)
    np.testing.assert_almost_equal(30.0, dip_30)
    np.testing.assert_almost_equal(90.0, rake_30)

    strike_10, dip_10, rake_10 = facet.source_builder.get_optimal_fault_plane(
        T, 10.0
    )
    np.testing.assert_almost_equal(0.0, strike_10)
    np.testing.assert_almost_equal(40.0, dip_10)
    np.testing.assert_almost_equal(90.0, rake_10)


test_get_optimal_fault_plane_contraction_1()


def test_get_optimal_fault_plane_strike_slip_1():
    T = np.array([[-1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.0]])

    strike, dip, rake = facet.source_builder.get_optimal_fault_plane(T, 30.0)

    np.testing.assert_almost_equal(300.0, strike)
    np.testing.assert_almost_equal(90.0, dip)


test_get_optimal_fault_plane_strike_slip_1()
