"""Unit tests for utils.py"""

import pytest

from utils import bearing, calc_travel, coordinate_distance, deg2rad, elevation


def test_deg2rad():
    """Unit tests for deg2rad()."""
    # Note: python's math package includes a radians function that
    # converts degrees to radians. This function could be eliminated
    # to reduce custom code.
    assert deg2rad(57.2958) == 1.0000003575641672
    assert deg2rad(1) == 0.017453292519943295
    assert deg2rad(-1) == -0.017453292519943295


@pytest.mark.skip(reason="Insufficient documentation to test. No docstrings.")
def test_elevation():
    """Unit test for elevation()."""
    pass


def test_bearing():
    """Unit test for bearing()."""
    # Example from: https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    lat1, long1 = 39.099912, -94.581213
    lat2, long2 = 38.627089, -90.200203
    expected_bearing = 96.51262423499941
    assert bearing(lat1, long1, lat2, long2) == expected_bearing


def test_coordinate_distance():
    """Unit test for coordinate_distance()."""
    # Used this app to calculate distance: https://www.movable-type.co.uk/scripts/latlong.html
    lat1, long1 = 39.099912, -94.581213
    lat2, long2 = 38.627089, -90.200203
    expected_distance = 382900.05037560174
    assert coordinate_distance(lat1, long1, lat2, long2) == expected_distance


@pytest.mark.skip(reason="Insufficient documentation to test. What is lead_s?")
def test_calc_travel():
    """Unit test for calc_travel()."""
    # note: the code in calc_travel is hard to understand because of the tangle
    # of calculations. consider reformatting and explaining or explore the possibility
    # using geopy
    pass
