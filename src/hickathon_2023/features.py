FEATURES = [
    "add_heat_generators_boiler",
    "add_heat_generators_heat_pump",
    "add_heat_generators_stove",
    "altitude",
    "area_code",
    "balcony_depth",
    "bearing_wall_agglomerate",
    "bearing_wall_bricks",
    "bearing_wall_chipboard",
    "bearing_wall_concrete",
    "bearing_wall_gritstone",
    "bearing_wall_indetermined",
    "bearing_wall_millstone",
    "bearing_wall_other",
    "bearing_wall_stone",
    "bearing_wall_wood",
    "building_category_condo",
    "building_category_house",
    "building_class_12_plus",
    "building_class_2_to_11",
    "building_class_indiv",
    "building_height_ft",
    "building_total_area_sqft",
    "building_type",
    "building_use_type_code",
    "building_year",
    # "clay_risk_level",
    "consumption_measurement_date",
    "has_air_conditioning",
    "heat_generators_boiler",
    "heat_generators_electric",
    "heat_generators_heat_pump",
    "heat_generators_solar",
    "heat_generators_stove",
    "heating_source_coal",
    "heating_source_electricity",
    "heating_source_gas",
    "heating_source_network",
    "heating_source_oil",
    "heating_source_wood",
    "heating_type",
    "is_crossing_building",
    "living_area_sqft",
    "lower_floor_material_brick",
    "lower_floor_material_concrete",
    "lower_floor_material_joist",
    "lower_floor_material_metal",
    "lower_floor_material_wood",
    "lowe_floor_thermal_conductivity",
    "lower_year_building",
    "main_heating_device",
    "main_heating_fuel",
    "main_water_heaters",
    "nb_commercial_units",
    "nb_dwellings",
    "nb_housing_units",
    "nb_parking_spaces",
    "number_of_fronts",
    "outer_wall_hollow",
    "outer_wall_thermal_conductivity",
    "outer_wall_thickness",
    "percentage_glazed_surfaced",
    "radon_risk_level",
    "renewable_energy_sources",
    "roof_material",
    "solar_heating",
    "solar_water_heating",
    "thermal_inertia",
    "upper_floor_LNC",
    "upper_floor_thermal_conductivity",
    "upper_year_building",
    "ventilation_type",
    "wall_insulation_external",
    "wall_insulation_insulated",
    "wall_insulation_internal",
    "wall_insulation_mob",
    "wall_insulation_reflection",
    "water_heater_boiler",
    "water_heater_heater",
    "water_heater_tank",
    "water_heating_coal",
    "water_heating_electricity",
    "water_heating_gas",
    "water_heating_network",
    "water_heating_oil",
    "water_heating_type",
    "water_heating_wood",
    "window_filling_type",
    "window_frame_material",
    "window_glazing_type",
    "window_heat_retention_factor",
    "window_orientation_east",
    "window_orientation_north",
    "window_orientation_south",
    "window_orientation_west",
    "window_thermal_conductivity",
]

FEATURES_ONEHOT = [
    "area_code",
    "building_type",
    "building_use_type_code",
    # "clay_risk_level",
    "heating_type",
    "lower_floor_adjacency_type",
    "lower_floor_insulation_type",
    "main_heating_device",
    "main_heating_fuel",
    "radon_risk_level",
    "roof_material",
    "thermal_inertia",
    "ventilation_type",
    "water_heating_type",
    "window_filling_type",
    "window_frame_material",
    "window_glazing_type",
]

FEATURES_DTYPES = {
    "balcony_depth": "np.int8",
    "consumption_measurement_date": "np.int16",
    "lower_year_building": "np.int16",
    "nb_parking_spaces": "np.int16",
    "number_of_fronts": "np.int8",
    "outer_wall_thickness": "np.int8",
    "radon_risk_level": "np.int8",
    "renewable_energy_sources": "np.int8",
    "thermal_inertia": "np.int8",
    "upper_year_building": "np.int16",
    "ventilation_type": "np.int8",
    "window_filling_type": "np.int8",
    "window_glazing_type": "np.int8",
}
