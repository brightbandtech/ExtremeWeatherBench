"""Tests for the calc module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import calc


class TestDerivedVariable:
    """Test the DerivedVariable base class."""

    def test_derived_variable_initialization(self):
        """Test that DerivedVariable initializes correctly."""

        # Create a concrete implementation for testing
        class TestDerivedVariable(calc.DerivedVariable):
            def compute(self, case, data, variables=None):
                return data

        dv = TestDerivedVariable("test_var", ["input1", "input2"])
        assert dv.name == "test_var"
        assert dv.input_variables == ["input1", "input2"]

    def test_check_variables_success(self):
        """Test that _check_variables passes with valid variables."""

        class TestDerivedVariable(calc.DerivedVariable):
            def compute(self, case, data, variables=None):
                return data

        dv = TestDerivedVariable("test_var", ["input1"])
        ds = xr.Dataset({"input1": xr.DataArray([1, 2, 3])})
        # Should not raise an exception
        dv._check_variables(ds, ["input1"])

    def test_check_variables_failure(self):
        """Test that _check_variables raises ValueError with invalid variables."""

        class TestDerivedVariable(calc.DerivedVariable):
            def compute(self, case, data, variables=None):
                return data

        dv = TestDerivedVariable("test_var", ["input1"])
        ds = xr.Dataset({"input1": xr.DataArray([1, 2, 3])})
        with pytest.raises(
            ValueError, match="Variable missing_var not found in dataset"
        ):
            dv._check_variables(ds, ["missing_var"])


class TestPracticallyPerfectHindcast:
    """Test the PracticallyPerfectHindcast derived variable."""

    def test_practically_perfect_hindcast_initialization(self):
        """Test that PracticallyPerfectHindcast initializes correctly."""
        pph = calc.PracticallyPerfectHindcast()
        assert pph.name == "practically_perfect_hindcast"
        assert pph.input_variables == ["report_type"]

    @patch("extremeweatherbench.calc.practically_perfect_hindcast")
    def test_practically_perfect_hindcast_compute(self, mock_pph):
        """Test that compute method calls the underlying function correctly."""
        pph = calc.PracticallyPerfectHindcast()
        mock_case = MagicMock()
        mock_case.location = MagicMock()
        mock_data = xr.Dataset({"report_type": xr.DataArray([1, 2, 3])})
        mock_result = xr.Dataset({"result": xr.DataArray([1, 2, 3])})
        mock_pph.return_value = mock_result

        result = pph.compute(mock_case, mock_data)

        mock_pph.assert_called_once_with(
            mock_data[["report_type"]],
            output_bounds=mock_case.location,
            report_type=["tor", "hail"],
        )
        assert result == mock_result


class TestCravenSignificantSevereParameter:
    """Test the CravenSignificantSevereParameter derived variable."""

    def test_craven_significant_severe_parameter_initialization(self):
        """Test that CravenSignificantSevereParameter initializes correctly."""
        cbss = calc.CravenSignificantSevereParameter()
        expected_vars = [
            "air_temperature",
            "dewpoint_temperature",
            "relative_humidity",
            "eastward_wind",
            "northward_wind",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]
        assert cbss.name == "craven_significant_severe_parameter"
        assert cbss.input_variables == expected_vars

    @patch("extremeweatherbench.calc.craven_brooks_sig_svr")
    def test_craven_significant_severe_parameter_compute(self, mock_cbss):
        """Test that compute method calls the underlying function correctly."""
        cbss = calc.CravenSignificantSevereParameter()
        mock_case = MagicMock()
        mock_data = xr.Dataset(
            {
                "air_temperature": xr.DataArray([1, 2, 3]),
                "dewpoint_temperature": xr.DataArray([1, 2, 3]),
                "relative_humidity": xr.DataArray([1, 2, 3]),
                "eastward_wind": xr.DataArray([1, 2, 3]),
                "northward_wind": xr.DataArray([1, 2, 3]),
                "surface_eastward_wind": xr.DataArray([1, 2, 3]),
                "surface_northward_wind": xr.DataArray([1, 2, 3]),
            }
        )
        mock_result = xr.Dataset({"result": xr.DataArray([1, 2, 3])})
        mock_cbss.return_value = mock_result

        result = cbss.compute(mock_case, mock_data)

        mock_cbss.assert_called_once_with(mock_data[cbss.input_variables])
        assert result == mock_result


# class TestPracticallyPerfectHindcastFunction:
#     """Test the practically_perfect_hindcast function."""

#     def test_practically_perfect_hindcast_basic(self):
#         """Test basic PPH calculation."""
#         # Create mock data
#         coords = pd.DataFrame(
#             {"latitude": [40.0, 40.25, 40.5], "longitude": [-100.0, -99.75, -99.5]}
#         )
#         ds = xr.Dataset(
#             {"reports": xr.DataArray([1, 1, 1], coords=[("index", range(3))])}
#         )

#         # Create a region
#         region = regions.BoundingBoxRegion(
#             latitude_min=39.5,
#             latitude_max=41.0,
#             longitude_min=-100.5,
#             longitude_max=-99.0,
#         )

#         result = calc.practically_perfect_hindcast(ds, region)

#         assert "practically_perfect" in result.data_vars
#         assert "reports" in result.data_vars
#         assert "latitude" in result.coords
#         assert "longitude" in result.coords

#     def test_practically_perfect_hindcast_no_reports(self):
#         """Test PPH calculation with no reports in region."""
#         coords = pd.DataFrame(
#             {
#                 "latitude": [50.0, 50.25, 50.5],  # Outside region
#                 "longitude": [-150.0, -149.75, -149.5],  # Outside region
#             }
#         )
#         ds = xr.Dataset(
#             {"reports": xr.DataArray([1, 1, 1], coords=[("index", range(3))])}
#         )

#         region = regions.BoundingBoxRegion(
#             latitude_min=39.5,
#             latitude_max=41.0,
#             longitude_min=-100.5,
#             longitude_max=-99.0,
#         )

#         result = calc.practically_perfect_hindcast(ds, region)

#         # Should still create the grid but with no reports
#         assert "practically_perfect" in result.data_vars
#         assert "reports" in result.data_vars
#         assert result.reports.sum() == 0


class TestDewpointCalculations:
    """Test dewpoint-related calculations."""

    def test_dewpoint_from_specific_humidity(self):
        """Test dewpoint calculation from specific humidity."""
        specific_humidity = xr.DataArray([0.01, 0.02, 0.03])  # kg/kg
        pressure = xr.DataArray([1000, 850, 700])  # hPa

        result = calc.dewpoint_from_specific_humidity(specific_humidity, pressure)

        assert isinstance(result, xr.DataArray)
        assert result.shape == specific_humidity.shape
        # Dewpoint should be reasonable values (typically -50 to 30 C)
        assert np.all(result > -60)
        assert np.all(result < 40)

    def test_dewpoint_from_vapor_pressure(self):
        """Test dewpoint calculation from vapor pressure."""
        vapor_pressure = xr.DataArray([10, 20, 30])  # hPa

        result = calc.dewpoint_from_vapor_pressure(vapor_pressure)

        assert isinstance(result, xr.DataArray)
        assert result.shape == vapor_pressure.shape
        # Higher vapor pressure should give higher dewpoint
        assert result[2] > result[1] > result[0]

    def test_saturation_vapor_pressure(self):
        """Test saturation vapor pressure calculation."""
        temperature = xr.DataArray([0, 10, 20, 30])  # Celsius

        result = calc.saturation_vapor_pressure(temperature)

        assert isinstance(result, xr.DataArray)
        assert result.shape == temperature.shape
        # Higher temperature should give higher saturation vapor pressure
        assert result[3] > result[2] > result[1] > result[0]
        # At 0C, should be close to 6.112 hPa
        assert np.isclose(result[0], 6.112, atol=0.1)


class TestMixingRatioCalculations:
    """Test mixing ratio calculations."""

    def test_mixing_ratio(self):
        """Test mixing ratio calculation."""
        partial_pressure = xr.DataArray([10, 20, 30])  # hPa
        total_pressure = xr.DataArray([1000, 1000, 1000])  # hPa

        result = calc.mixing_ratio(partial_pressure, total_pressure)

        assert isinstance(result, xr.DataArray)
        assert result.shape == partial_pressure.shape
        # Higher partial pressure should give higher mixing ratio
        assert result[2] > result[1] > result[0]

    def test_vapor_pressure(self):
        """Test vapor pressure calculation."""
        pressure = xr.DataArray([1000, 850, 700])  # hPa
        mixing_ratio = xr.DataArray([0.01, 0.02, 0.03])  # kg/kg

        result = calc.vapor_pressure(pressure, mixing_ratio)

        assert isinstance(result, xr.DataArray)
        assert result.shape == pressure.shape


class TestTemperatureCalculations:
    """Test temperature-related calculations."""

    def test_potential_temperature_celsius(self):
        """Test potential temperature calculation with Celsius input."""
        temperature = xr.DataArray([0, 10, 20])  # Celsius
        pressure = xr.DataArray([1000, 850, 700])  # hPa

        result = calc.potential_temperature(temperature, pressure, units="C")

        assert isinstance(result, xr.DataArray)
        assert result.shape == temperature.shape
        # Potential temperature should be higher than actual temperature
        assert np.all(result > temperature + 273.15)

    def test_potential_temperature_kelvin(self):
        """Test potential temperature calculation with Kelvin input."""
        temperature = xr.DataArray([273.15, 283.15, 293.15])  # Kelvin
        pressure = xr.DataArray([1000, 850, 700])  # hPa

        result = calc.potential_temperature(temperature, pressure, units="K")

        assert isinstance(result, xr.DataArray)
        assert result.shape == temperature.shape

    def test_potential_temperature_invalid_units(self):
        """Test potential temperature with invalid units."""
        temperature = xr.DataArray([0, 10, 20])
        pressure = xr.DataArray([1000, 850, 700])

        with pytest.raises(ValueError, match="Unknown units: F"):
            calc.potential_temperature(temperature, pressure, units="F")

    def test_dry_lapse_1d(self):
        """Test dry lapse calculation with 1D arrays."""
        pressure = xr.DataArray([1000, 850, 700])  # hPa
        temperature = xr.DataArray([20])  # Celsius

        result = calc.dry_lapse(pressure, temperature)

        assert isinstance(result, xr.DataArray)
        assert result.shape == pressure.shape
        # Temperature should decrease with decreasing pressure
        assert result[0] > result[1] > result[2]

    def test_dry_lapse_2d(self):
        """Test dry lapse calculation with 2D arrays."""
        pressure = xr.DataArray([[1000, 850, 700], [1000, 850, 700]])  # hPa
        temperature = xr.DataArray([[20, 20, 20], [25, 25, 25]])  # Celsius

        result = calc.dry_lapse(pressure, temperature)

        assert isinstance(result, xr.DataArray)
        assert result.shape == pressure.shape


class TestVirtualTemperatureCalculations:
    """Test virtual temperature calculations."""

    def test_virtual_temperature(self):
        """Test virtual temperature calculation."""
        temperature = xr.DataArray([273.15, 283.15, 293.15])  # Kelvin
        mixing_ratio = xr.DataArray([0.01, 0.02, 0.03])  # kg/kg

        result = calc.virtual_temperature(temperature, mixing_ratio)

        assert isinstance(result, xr.DataArray)
        assert result.shape == temperature.shape
        # Virtual temperature should be higher than actual temperature
        assert np.all(result > temperature)

    def test_virtual_temperature_from_dewpoint(self):
        """Test virtual temperature calculation from dewpoint."""
        pressure = xr.DataArray([1000, 850, 700])  # hPa
        temperature = xr.DataArray([20, 15, 10])  # Celsius
        dewpoint = xr.DataArray([15, 10, 5])  # Celsius

        result = calc.virtual_temperature_from_dewpoint(pressure, temperature, dewpoint)

        assert isinstance(result, xr.DataArray)
        assert result.shape == temperature.shape


class TestPressureHeightCalculations:
    """Test pressure and height calculations."""

    def test_get_pressure_height(self):
        """Test pressure height calculation."""
        pressure = xr.DataArray([1000, 850, 700, 500])  # hPa

        result_pressure, result_height = calc.get_pressure_height(pressure)

        assert isinstance(result_pressure, np.ndarray)
        assert isinstance(result_height, np.ndarray)
        assert result_pressure.shape == pressure.shape
        assert result_height.shape == pressure.shape
        # Height should increase with decreasing pressure
        assert result_height[0] < result_height[1] < result_height[2] < result_height[3]

    def test_exner_function(self):
        """Test Exner function calculation."""
        pressure = xr.DataArray([1000, 850, 700, 500])  # hPa

        result = calc.exner_function(pressure)

        assert isinstance(result, xr.DataArray)
        assert result.shape == pressure.shape
        # Exner function should decrease with decreasing pressure
        assert result[0] > result[1] > result[2] > result[3]


class TestSaturationCalculations:
    """Test saturation-related calculations."""

    def test_saturation_mixing_ratio(self):
        """Test saturation mixing ratio calculation."""
        pressure = xr.DataArray([1000, 850, 700])  # hPa
        temperature = xr.DataArray([20, 15, 10])  # Celsius

        result = calc.saturation_mixing_ratio(pressure, temperature)

        assert isinstance(result, xr.DataArray)
        assert result.shape == temperature.shape
        # Higher temperature should give higher saturation mixing ratio
        assert result[0] > result[1] > result[2]


class TestLCLCalculations:
    """Test LCL (Lifting Condensation Level) calculations."""

    def test_new_lcl(self):
        """Test LCL calculation."""
        pressure_prof = xr.DataArray([1000])  # hPa
        temp_prof = xr.DataArray([20 + 273.15])  # Kelvin
        dew_prof = xr.DataArray([15 + 273.15])  # Kelvin

        lcl_pressure, lcl_temp = calc.new_lcl(pressure_prof, temp_prof, dew_prof)

        assert isinstance(lcl_pressure, xr.DataArray)
        assert isinstance(lcl_temp, xr.DataArray)
        # LCL pressure should be less than surface pressure
        assert lcl_pressure < pressure_prof

    def test_insert_lcl_level_fast(self):
        """Test fast LCL level insertion."""
        pressure = xr.DataArray([[1000, 850, 700]])  # hPa
        temperature = xr.DataArray([[20, 15, 10]])  # Celsius
        lcl_pressure = xr.DataArray([[900]])  # hPa

        result = calc.insert_lcl_level_fast(pressure, temperature, lcl_pressure)

        assert isinstance(result, np.ndarray)
        # Should have one more level than original
        assert result.shape[-1] == pressure.shape[-1] + 1


class TestInterpolationFunctions:
    """Test interpolation functions."""

    def test_log_interpolate(self):
        """Test logarithmic interpolation."""
        x = xr.DataArray([900, 800, 700])  # Target pressures
        xp = xr.DataArray([1000, 850, 700])  # Source pressures
        var = xr.DataArray([20, 15, 10])  # Temperature values

        result = calc.log_interpolate(x, xp, var)

        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape

    def test_combine_profiles(self):
        """Test profile combination."""
        press_lower = xr.DataArray([1000, 950])
        lcl_p = xr.DataArray([900])
        press_upper = xr.DataArray([850, 800])
        temp_lower = xr.DataArray([20, 18])
        lcl_td = xr.DataArray([15])
        temp_upper = xr.DataArray([12, 8])

        result_pressure, result_temp = calc.combine_profiles(
            press_lower, lcl_p, press_upper, temp_lower, lcl_td, temp_upper
        )

        assert isinstance(result_pressure, np.ndarray)
        assert isinstance(result_temp, np.ndarray)
        # Should have combined length
        expected_length = len(press_lower) + len(lcl_p) + len(press_upper)
        assert len(result_pressure) == expected_length


class TestIntersectionFunctions:
    """Test intersection finding functions."""

    def test_find_intersection_simple(self):
        """Test simple intersection finding."""
        x = np.array([1, 2, 3, 4, 5])
        y1 = np.array([1, 2, 3, 4, 5])
        y2 = np.array([5, 4, 3, 2, 1])

        x_intersect, y_intersect = calc.find_intersection(x, y1, y2)

        # Should find intersection at x=3, y=3
        assert x_intersect is not None
        assert y_intersect is not None
        assert np.isclose(x_intersect[0], 3)
        assert np.isclose(y_intersect[0], 3)

    def test_find_intersection_no_intersection(self):
        """Test intersection finding with no intersection."""
        x = np.array([1, 2, 3, 4, 5])
        y1 = np.array([1, 2, 3, 4, 5])
        y2 = np.array([6, 7, 8, 9, 10])

        x_intersect, y_intersect = calc.find_intersection(x, y1, y2)

        assert x_intersect is None
        assert y_intersect is None

    def test_find_intersections(self):
        """Test multiple intersection finding."""
        x = np.array([1000, 850, 700, 500])
        y1 = np.array([20, 15, 10, 5])  # Parcel temperature
        y2 = np.array([15, 12, 8, 2])  # Environment temperature

        x_intersect, y_intersect = calc.find_intersections(x, y1, y2)

        assert isinstance(x_intersect, np.ndarray)
        assert isinstance(y_intersect, np.ndarray)


class TestCAPECINCalculations:
    """Test CAPE and CIN calculations."""

    def test_mlcape_cin_basic(self):
        """Test basic CAPE/CIN calculation."""
        pressure = np.array([[1000, 850, 700, 500]])
        temperature = np.array([[20, 15, 10, 5]])  # Celsius
        dewpoint = np.array([[15, 10, 5, 0]])  # Celsius
        parcel_profile = np.array([[20, 18, 15, 12]])  # Celsius

        cape, cin = calc.mlcape_cin(pressure, temperature, dewpoint, parcel_profile)

        assert isinstance(cape, np.ndarray)
        assert isinstance(cin, np.ndarray)
        assert cape.shape == pressure.shape[:-1]
        assert cin.shape == pressure.shape[:-1]
        # CAPE should be non-negative
        assert np.all(cape >= 0)
        # CIN should be non-positive
        assert np.all(cin <= 0)

    def test_equilibrium_level(self):
        """Test equilibrium level calculation."""
        pressure = np.array([1000, 850, 700, 500])
        temperature = np.array([20, 15, 10, 5])  # Celsius
        dewpoint = np.array([15, 10, 5, 0])  # Celsius
        parcel_profile = np.array([20, 18, 15, 12])  # Celsius

        el_pressure, el_temp = calc.equilibrium_level(
            pressure, temperature, dewpoint, parcel_profile
        )

        assert isinstance(el_pressure, (float, np.ndarray))
        assert isinstance(el_temp, (float, np.ndarray))

    def test_level_free_convection(self):
        """Test level of free convection calculation."""
        pressure = np.array([1000, 850, 700, 500])
        temperature = np.array([20, 15, 10, 5])  # Celsius
        dewpoint = np.array([15, 10, 5, 0])  # Celsius
        parcel_profile = np.array([20, 18, 15, 12])  # Celsius

        lfc_pressure, lfc_temp = calc.level_free_convection(
            pressure, temperature, dewpoint, parcel_profile
        )

        assert isinstance(lfc_pressure, (float, np.ndarray))
        assert isinstance(lfc_temp, (float, np.ndarray))


class TestUtilityFunctions:
    """Test utility functions."""

    def test_next_non_masked_element(self):
        """Test next non-masked element function."""
        # Test with regular array
        a = np.array([1, 2, 3, 4, 5])
        idx, val = calc._next_non_masked_element(a, 2)
        assert idx == 2
        assert val == 3

    def test_basic_ds_checks(self):
        """Test basic dataset checks."""
        # Create dataset with ascending pressure levels
        ds = xr.Dataset(
            {
                "level": xr.DataArray([500, 700, 850, 1000]),
                "temperature": xr.DataArray([[20, 15, 10, 5]]),
                "pressure": xr.DataArray([[500, 700, 850, 1000]]),
            }
        )

        result = calc._basic_ds_checks(ds)

        # Should sort pressure levels in descending order
        assert result["level"][0] > result["level"][-1]

    @patch("extremeweatherbench.calc.resources.files")
    def test_load_moist_lapse_lookup(self, mock_files):
        """Test loading of moist lapse lookup table."""
        mock_path = MagicMock()
        mock_files.return_value.joinpath.return_value = mock_path

        with patch("pandas.read_parquet") as mock_read:
            mock_read.return_value = pd.DataFrame({"col1": [1, 2, 3]})
            result = calc.load_moist_lapse_lookup()

            assert isinstance(result, pd.DataFrame)
            mock_read.assert_called_once_with(mock_path)


class TestCravenBrooksFunctions:
    """Test Craven-Brooks related functions."""

    def test_craven_brooks_sig_svr(self):
        """Test Craven-Brooks significant severe parameter calculation."""
        ds = xr.Dataset(
            {
                "level": xr.DataArray([1000, 850, 700, 500]),
                "air_temperature": xr.DataArray([[20, 15, 10, 5]]),
                "dewpoint_temperature": xr.DataArray([[15, 10, 5, 0]]),
                "relative_humidity": xr.DataArray([[80, 70, 60, 50]]),
                "eastward_wind": xr.DataArray([[10, 15, 20, 25]]),
                "northward_wind": xr.DataArray([[5, 10, 15, 20]]),
                "surface_eastward_wind": xr.DataArray([[5]]),
                "surface_northward_wind": xr.DataArray([[2]]),
            }
        )

        variable_mapping = {
            "pressure": "level",
            "temperature": "air_temperature",
            "dewpoint": "dewpoint_temperature",
            "eastward_wind": "eastward_wind",
            "northward_wind": "northward_wind",
            "surface_eastward_wind": "surface_eastward_wind",
            "surface_northward_wind": "surface_northward_wind",
        }

        with patch("extremeweatherbench.calc.mixed_layer_cape_cin") as mock_cape:
            with patch("extremeweatherbench.calc.low_level_shear") as mock_shear:
                mock_cape.return_value = (xr.DataArray([1000]), xr.DataArray([-100]))
                mock_shear.return_value = xr.DataArray([20])

                result = calc.craven_brooks_sig_svr(ds, variable_mapping)

                assert isinstance(result, xr.DataArray)
                # CBSS should be CAPE * shear
                assert result.values[0] == 20000

    def test_low_level_shear(self):
        """Test low level shear calculation."""
        ds = xr.Dataset(
            {
                "level": xr.DataArray([1000, 850, 700, 500]),
                "eastward_wind": xr.DataArray([[10, 15, 20, 25]]),
                "northward_wind": xr.DataArray([[5, 10, 15, 20]]),
                "surface_eastward_wind": xr.DataArray([[5]]),
                "surface_northward_wind": xr.DataArray([[2]]),
            }
        )

        variable_mapping = {
            "eastward_wind": "eastward_wind",
            "northward_wind": "northward_wind",
            "surface_eastward_wind": "surface_eastward_wind",
            "surface_northward_wind": "surface_northward_wind",
        }

        result = calc.low_level_shear(ds, variable_mapping)

        assert isinstance(result, xr.DataArray)
        # Should calculate shear between 500mb and surface
        expected_shear = np.sqrt((25 - 5) ** 2 + (20 - 2) ** 2)
        assert np.isclose(result.values[0], expected_shear)


class TestMoistLapseFunctions:
    """Test moist lapse related functions."""

    def test_moist_lapse_lookup(self):
        """Test moist lapse lookup function."""
        target_pressure = np.array([850, 700, 500])
        target_temp = np.array([15, 10, 5])
        reference_pressure = np.array([1000, 1000, 1000])

        with patch("extremeweatherbench.calc.load_moist_lapse_lookup") as mock_load:
            # Create mock lookup table
            mock_df = pd.DataFrame(
                {
                    1000: [20, 15, 10, 5],
                    850: [18, 13, 8, 3],
                    700: [16, 11, 6, 1],
                    500: [14, 9, 4, -1],
                },
                index=[1000, 850, 700, 500],
            )
            mock_load.return_value = mock_df

            result = calc.moist_lapse_lookup(
                target_pressure, target_temp, reference_pressure
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == target_pressure.shape


class TestMixedParcelFunctions:
    """Test mixed parcel calculations."""

    def test_mixed_parcel(self):
        """Test mixed parcel calculation."""
        ds = xr.Dataset(
            {
                "level": xr.DataArray([1000, 850, 700, 500]),
                "air_temperature": xr.DataArray([[20, 15, 10, 5]]),
                "dewpoint_temperature": xr.DataArray([[15, 10, 5, 0]]),
                "pressure": xr.DataArray([[1000, 850, 700, 500]]),
            }
        )

        variable_mapping = {
            "temperature": "air_temperature",
            "dewpoint": "dewpoint_temperature",
            "pressure": "pressure",
        }

        start_pressure, parcel_temp, parcel_dewpoint = calc.mixed_parcel(
            ds, variable_mapping, depth=100
        )

        assert isinstance(start_pressure, xr.DataArray)
        assert isinstance(parcel_temp, xr.DataArray)
        assert isinstance(parcel_dewpoint, xr.DataArray)
        assert start_pressure.values[0] == 1000  # Should be surface pressure


class TestIntegrationFunctions:
    """Test integration functions."""

    def test_interp_integrate(self):
        """Test interpolation and integration function."""
        pressure = xr.DataArray([1000, 850, 700, 500])
        pressure_interp = xr.DataArray([1000, 900, 800, 700, 600, 500])
        layer_depth = 500  # hPa
        vars_data = xr.DataArray([20, 15, 10, 5])

        result = calc._interp_integrate(
            pressure, pressure_interp, layer_depth, vars_data
        )

        assert isinstance(result, xr.DataArray)
        # Should return integrated value divided by layer depth
        assert result.shape == vars_data.shape[:-1]


if __name__ == "__main__":
    pytest.main([__file__])
