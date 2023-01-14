from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def compute_year(df):
    df["year"] = pd.to_datetime(df["consumption_measurement_date"]).dt.year
    return df


def processing_bearing_wall(df):
    df["bearing_wall_concrete"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "CONC" in x
    )
    df["bearing_wall_wood"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "WOOD" in x
    )
    df["bearing_wall_bricks"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "BRIC" in x
    )
    df["bearing_wall_gritstone"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "GRIT" in x
    )
    df["bearing_wall_stone"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "STON" in x and "GRIT" not in x
    )
    df["bearing_wall_indetermined"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "INDE" in x
    )
    df["bearing_wall_agglomerate"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "AGGL" in x
    )
    df["bearing_wall_millstone"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "MILL" in x
    )
    df["bearing_wall_chipboard"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "CHIP" in x
    )
    df["bearing_wall_other"] = df["bearing_wall_material"].apply(
        lambda x: type(x) != float and "OTHE" in x
    )
    return df.drop("bearing_wall_material", axis=1)


def processing_main_heating_type(df):
    def aux_device(x):
        if "boil" in x:
            return "bo"
        elif "pump" in x:
            return "hp"
        elif "stove" in x:
            return "st"
        elif "radia" in x:
            return "ra"
        elif "joule" in x:
            return "jo"
        else:
            return "ot"

    def aux_fuel(x):
        if "gas" in x or "butane" in x:
            return "gas"
        elif "oil " in x:
            return "oil"
        elif "solar" in x:
            return "sol"
        elif "wood" in x:
            return "woo"
        elif "coal" in x or "charb" in x:
            return "coa"
        else:
            return "oth"

    df["main_heating_device"] = df["main_heat_generators"].apply(aux_device)
    df["main_heating_fuel"] = df["main_heat_generators"].apply(aux_fuel)
    return df.drop("main_heating_generators", axis=1)


def processing_balcony_depth(df):
    values = {np.nan: 0, "< 1 m": 1, "1 <= … < 2": 2, "2 <= … < 3": 3, "3 <=": 4}
    df["balcony_depth"] = df["balcony_depth"].map(values)
    return df


def processing_window_glazing_type(df):
    values = {
        np.nan: 0,
        "single glazing": 1,
        "double glazing": 2,
        "triple glazing": 3,
        "glass block or polycarbonate": 3,
        "overglazing": 3,
    }
    df["window_glazing_type"] = df["window_glazing_type"].map(values)
    return df


def processing_window_frame_material(df):
    def aux(x):
        materials = ["pvc", "wood", "metal", "with thermal", "poly", "glass"]
        ret, i = False, 0
        while i < len(materials) and (not ret or ret == "metal"):
            if type(x) != float and materials[i] in x:
                ret = i
            i += 1
        return ret or -1

    df["window_frame_material"] = df["window_frame_material"].apply(aux)
    return df


def processing_window_filling_type(df):
    values = {np.nan: 0, "dry air": 1, "argon or krypton": 2}
    df["window_filling_type"] = df["window_filling_type"].map(values)
    return df


def processing_water_heating_type(df):
    values = {
        np.nan: 0,
        "individual": 0,
        "collective": 1,
    }  # fill nan with the most frequent class
    df["water_heating_type"] = df["water_heating_type"].map(values)
    return df


def processing_heating_type(df):
    values = {
        np.nan: 0,
        "individual": 0,
        "collective": 1,
    }  # fill nan with the most frequent class
    df["heating_type"] = df["heating_type"].map(values)
    return df


def processing_water_heaters(df):
    df["water_heater_boiler"] = df["water_heaters"].apply(lambda x: x.count("boiler"))
    df["water_heater_tank"] = df["water_heaters"].apply(lambda x: x.count("tank"))
    df["water_heater_heater"] = df["water_heaters"].apply(lambda x: x.count("heater"))
    return df.drop("water_heaters", axis=1)


def processing_water_heating_energy_source(df):
    df["water_heating_oil"] = df["water_heating_energy_source"].apply(
        lambda x: type(x) != float and "oil" in x
    )
    df["water_heating_gas"] = df["water_heating_energy_source"].apply(
        lambda x: type(x) != float and ("gas" in x or "butane" in x)
    )
    df["water_heating_electricity"] = df["water_heating_energy_source"].apply(
        lambda x: type(x) != float and "elec" in x
    )
    df["water_heating_wood"] = df["water_heating_energy_source"].apply(
        lambda x: type(x) != float and "wood" in x
    )
    df["water_heating_network"] = df["water_heating_energy_source"].apply(
        lambda x: type(x) != float and ("network" in x or "reseau" in x)
    )
    df["water_heating_coal"] = df["water_heating_energy_source"].apply(
        lambda x: type(x) != float and "coal" in x
    )
    return df


def processing_ventilation_type(df):
    values = {
        "humidity sensitive mechanical gas ventilation": 0,
        "Double flow mechanical ventilation with exchanger": 0,
        "Humidity sensitive mechanical exhaust ventilation and air inlets": 0,
        "Hybrid ventilation with humidity sensitive air inlets": 0,
        "Humidity sensitive mechanical exhaust ventilation": 1,
        "Canadian well": 1,
        "Double flow mechanical ventilation without exchanger": 1,
        "Self-regulating mechanical ventilation after 1982": 2,
        "Natural ventilation with humidity sensitive air inlets": 2,
        "Hybrid ventilation": 2,
        "Mechanical extractor on unmodified existing natural ventilation duct": 3,
        "Self-regulating mechanical ventilation before 1982": 4,
        "Ventilation by opening windows": 5,
        "Natural ventilation by duct": 6,
        "Ventilation system with high and low air inlets": 7,
    }
    df["ventilation_type"] = df["ventilation_type"].map(values)
    return df


def processing_wall_insulation_type(df):
    df["wall_insulation_mob"] = df["wall_insulation_type"].apply(
        lambda x: type(x) != float and "MOB" in x
    )
    df["wall_insulation_external"] = df["wall_insulation_type"].apply(
        lambda x: type(x) != float and "exter" in x
    )
    df["wall_insulation_internal"] = df["wall_insulation_type"].apply(
        lambda x: type(x) != float and "refle" in x
    )
    df["wall_insulation_reflection"] = df["wall_insulation_type"].apply(
        lambda x: type(x) != float and "inter" in x
    )
    df["wall_insulation_insulated"] = df["wall_insulation_type"].apply(
        lambda x: type(x) != float and "insul" in x
    )
    return df.drop("wall_insulation_type", axis=1)


def processing_building_category(df):
    df["building_category_condo"] = df["building_category"].apply(
        lambda x: x.count("condo")
    )
    df["building_category_house"] = df["building_category"].apply(
        lambda x: x.count("house")
    )
    return df.drop("building_category", axis=1)


def processing_building_class(df):
    df["building_class_indiv"] = df["building_class"].apply(lambda x: x.count("in"))
    df["building_class_2_to_11"] = df["building_class"].apply(lambda x: x.count("11"))
    df["building_class_12_plus"] = df["building_class"].apply(lambda x: x.count("12"))
    return df.drop("building_class", axis=1)


def processing_heating_energy_source(df):
    df["heat_source_oil"] = df["heating_energy_source"].apply(
        lambda x: type(x) != float and ("oil" in x or "fuel" in x or "fioul" in x)
    )
    df["heat_source_gas"] = df["heating_energy_source"].apply(
        lambda x: type(x) != float and ("gas" in x or "butane" in x)
    )
    df["heat_source_electricity"] = df["heating_energy_source"].apply(
        lambda x: type(x) != float and "elec" in x
    )
    df["heat_source_wood"] = df["heating_energy_source"].apply(
        lambda x: type(x) != float and "wood" in x
    )
    df["heat_source_network"] = df["heating_energy_source"].apply(
        lambda x: type(x) != float and ("network" in x or "reseau" in x)
    )
    df["heat_source_coal"] = df["heating_energy_source"].apply(
        lambda x: type(x) != float and ("coal" in x or "charbon" in x)
    )
    return df.drop("heating_energy_source", axis=1)


def processing_heat_generators(df):
    df["heat_generators_boiler"] = df["heat_generators"].apply(
        lambda x: x.count("boiler")
    )
    df["heat_generators_stove"] = df["heat_generators"].apply(
        lambda x: x.count("stove")
    )
    df["heat_generators_solar"] = df["heat_generators"].apply(
        lambda x: x.count("solar")
    )
    df["heat_generators_electric"] = df["heat_generators"].apply(
        lambda x: x.count("electric")
    )
    df["heat_generators_heat_pump"] = df["heat_generators"].apply(
        lambda x: x.count("pump")
    )
    return df.drop("heat_generators", axis=1)


def processing_additional_heat_generators(df):
    df["add_heat_generators_boiler"] = df["additional_heat_generators"].apply(
        lambda x: "boiler" in x
    )
    df["add_heat_generators_stove"] = df["additional_heat_generators"].apply(
        lambda x: "stove" in x
    )
    df["add_heat_generators_heat_pump"] = df["additional_heat_generators"].apply(
        lambda x: "pump" in x
    )
    return df.drop("additional_heat_generators", axis=1)


def processing_additional_water_heaters(df):
    df["add_water_heater_electric"] = df["additional_water_heaters"].apply(
        lambda x: type(x) != float and "electric" in x
    )
    df["add_water_heater_gas"] = df["additional_water_heaters"].apply(
        lambda x: type(x) != float and ("gas" in x or "gaz" in x or "butane" in x)
    )
    df["add_water_heater_oil"] = df["additional_water_heaters"].apply(
        lambda x: type(x) != float and "oil" in x
    )
    return df.drop("additional_water_heaters", axis=1)


def processing_renewable_energy_sources(df):
    df["renewable_energy_sources"] = df["renewable_energy_sources"].apply(
        lambda x: 1 + x.count("+") if type(x) != float else 0
    )
    return df


def processing_crossing_building(df):
    values_cross = {
        "crossing east west": 0,
        "through all way": 0,
        "crossing north south": 0,
        "nan": "nan",
        "through 90°": 0,
        "not through": 0,
        "all through crossing (weak)": 1,
        "east-west crossing (weak)": 1,
        "90° crossing (weak)": 1,
        "north-south crossing (weak)": 1,
    }
    values_fronts = {
        "crossing east west": 2,
        "through all way": 4,
        "crossing north south": 2,
        "nan": 0,
        "through 90°": 2,
        "not through": 1,
        "all through crossing (weak)": 4,
        "east-west crossing (weak)": 2,
        "90° crossing (weak)": 2,
        "north-south crossing (weak)": 2,
    }

    df["number_of_fronts"] = df["is_crossing_building"].map(values_fronts)
    df["is_crossing_building"] = df["is_crossing_building"].map(values_cross)
    return df


def processing_consumption_measurement_date(df):
    zero = datetime.strptime(df["consumption_measurement_date"].min(), "%Y-%m-%d")
    one = datetime.strptime(df["consumption_measurement_date"].max(), "%Y-%m-%d")

    def aux(x):
        date = datetime.strptime(x, "%Y-%m-%d")
        return (date - zero).days / (one - zero).days

    df["consumption_measurement_date"] = df["consumption_measurement_date"].apply(aux)
    return df


def processing_outer_wall_materials(df):
    df["outer_wall_hollow"] = df["outer_wall_materials"].apply(
        lambda x: type(x) != float and "hollow" in x
    )
    return df


def processing_years(df):
    dic_year_lower = {
        "1949-1970": 1949,
        "1970-1988": 1970,
        "1989-1999": 1989,
        "2000-2005": 2000,
        "2006-2012": 2006,
        "<1948": np.nan,
        ">2012": 2012,
        "bad sup": np.nan,
    }
    dic_year_upper = {
        "1949-1970": 1970,
        "1970-1988": 1988,
        "1989-1999": 1999,
        "2000-2005": 2005,
        "2006-2012": 2012,
        "<1948": 1948,
        ">2012": 2020,
        "bad sup": np.nan,
    }
    df["lower_year_building"] = df["building_period"].map(dic_year_lower)
    df["upper_year_building"] = df["building_period"].map(dic_year_upper)
    df["building_year"] = (
        df["building_year"]
        .fillna(df["lower_year_building"])
        .fillna(df["upper_year_building"])
        .fillna(df["building_year"].mean())
    )
    df["lower_year_building"] = df["lower_year_building"].fillna(df["building_year"])
    df["upper_year_building"] = df["upper_year_building"].fillna(df["building_year"])
    return df


def processing_upper_conductivity(df):
    df["upper_floor_thermal_conductivity"] = df.lowe_floor_thermal_conductivity.fillna(
        df.groupby(["upper_floor_insulation_type"])[
            "upper_floor_thermal_conductivity"
        ].transform("median")
    )
    df["upper_floor_thermal_conductivity"] = df.lowe_floor_thermal_conductivity.fillna(
        df.groupby(["upper_floor_material"])[
            "upper_floor_thermal_conductivity"
        ].transform("median")
    )
    df["upper_floor_thermal_conductivity"] = df[
        "upper_floor_thermal_conductivity"
    ].fillna(df["upper_floor_thermal_conductivity"].median())
    return df


def processing_lower_conductivity(df):
    df["lowe_floor_thermal_conductivity"] = df.lowe_floor_thermal_conductivity.fillna(
        df.groupby(["lower_floor_insulation_type"])[
            "lowe_floor_thermal_conductivity"
        ].transform("median")
    )
    df["lowe_floor_thermal_conductivity"] = df.lowe_floor_thermal_conductivity.fillna(
        df.groupby(["lower_floor_material"])[
            "lowe_floor_thermal_conductivity"
        ].transform("median")
    )
    df["lowe_floor_thermal_conductivity"] = df[
        "lowe_floor_thermal_conductivity"
    ].fillna(df["lowe_floor_thermal_conductivity"].median())
    return df


def processing_thermal_inertia(df):
    dic_inertia = {"low": 1, "medium": 2, "high": 3, "very high": 4}
    df["thermal_inertia"] = df["thermal_inertia"].map(dic_inertia).fillna(1)
    return df


def processing_outer_thickness(df):
    def get_thickness(thickness):
        try:
            return int(thickness[:2])
        except ValueError:
            return 24.4

    mean_thickness = str(
        df["outer_wall_thickness"].dropna().apply(get_thickness).mean()
    )
    df["outer_wall_thickness"] = (
        df["outer_wall_thickness"].fillna(mean_thickness).apply(get_thickness)
    )
    return df


def processing_upper_floor_adjacency_type(df):
    df["upper_floor_LNC"] = df["upper_floor_adjacency_type"] == "LNC"
    return df


def processing_radon(df):
    dic_radon = {"low": 0, "medium": 1, "high": 2}
    df["radon_risk_level"] = df["radon_risk_level"].map(dic_radon).fillna(1)
    return df


def processing_window_orientation(df):
    df["north_window"] = df["window_orientation"].apply(
        lambda x: type(x) != float and ("north" in x or "nord" in x)
    )
    df["east_window"] = df["window_orientation"].apply(
        lambda x: type(x) != float and ("east" in x or "est" in x)
    )
    df["south_window"] = df["window_orientation"].apply(
        lambda x: type(x) != float and ("south" in x or "sud" in x)
    )
    df["west_window"] = df["window_orientation"].apply(
        lambda x: type(x) != float and ("west" in x or "ouest" in x)
    )
    return df.drop("window_orientation", axis=1)


def processing_nb_parking_spaces(df):
    df["nb_parking_spaces"] = df.mask(
        (df["building_type"] == "House") & (df["nb_parking_spaces"] > 4), 4
    )
    return df


def processing_roof_materials(df):
    values = {
        "TILES - ZINC ALUMINUM": "TILES",
        "TILES - OTHERS": "TILES",
        "CONCRETE - TILES": "CONCRETE",
        "SLATES - ZINC ALUMINUM": "SLATE",
        "CONCRETE - OTHERS": "CONCRETE",
        "SLATE - TILES": "SLATE",
        "SLATE - OTHERS": "SLATE",
        "ZINC ALUMINUM - OTHERS": "ZINC",
        "CONCRETE - ZINC ALUMINUM": "CONCRETE",
        "SLATE - CONCRETE": "SLATE",
        "INDETERMINATE": "OTHERS",
    }
    df["roof_material"] = df["roof_material"].map(values).fillna("OTHERS")
    return df


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        processing_functions_list = [
            compute_year,
            processing_bearing_wall,
            processing_main_heating_type,
            processing_balcony_depth,
            processing_window_glazing_type,
            processing_window_frame_material,
            processing_window_filling_type,
            processing_water_heating_type,
            processing_heating_type,
            processing_water_heaters,
            processing_heating_energy_source,
            processing_water_heating_energy_source,
            processing_ventilation_type,
            processing_wall_insulation_type,
            processing_building_category,
            processing_building_class,
            processing_heat_generators,
            processing_additional_heat_generators,
            processing_additional_water_heaters,
            processing_renewable_energy_sources,
            processing_crossing_building,
            processing_consumption_measurement_date,
            processing_outer_wall_materials,
            processing_years,
            processing_upper_conductivity,
            processing_lower_conductivity,
            processing_thermal_inertia,
            processing_outer_thickness,
            processing_upper_floor_adjacency_type,
            processing_radon,
            processing_window_orientation,
            processing_nb_parking_spaces,
        ]
        X = X.copy()
        for function in processing_functions_list:
            X = function(X)
        return X
