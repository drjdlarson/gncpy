import pytest
import numpy.testing as test

import gncpy.sensors as sense


@pytest.mark.incremental
class TestGPSConstellation:
    def test_parse_almanac(self, alm_file):
        const = sense.GPSConstellation()
        const.parse_almanac(alm_file)

        assert "2" in const.sats
        assert "11" in const.sats

        test.assert_almost_equal(const.sats["2"].inc,
                                 0.9596631746)
        test.assert_almost_equal(const.sats["11"].ascen_rate,
                                 float(-0.8526069431E-008))

    def test_propagate(self, alm_file):
        const = sense.GPSConstellation()
        const.parse_almanac(alm_file)

        assert const.sats["2"].true_pos_ECEF.size == 0

        const.propagate(const.sats["2"].toe + 1)

        assert const.sats["2"].true_pos_ECEF.size == 3
