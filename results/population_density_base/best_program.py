def estimator(image):
    building_mask = segment(image, "buildings")
    road_mask = segment(image, "roads")
    residential_building_mask = segment(image, "residential building")
    park_mask = segment(image, "park")
    forest_mask = segment(image, "forest")
    water_mask = segment(image, "water")
    highway_mask = segment(image, "highway")
    non_residential_building_mask = segment(image, "non-residential buildings")
    beach_mask = segment(image, "beach")
    lake_mask = segment(image, "lake")
    river_mask = segment(image, "river")

    building_pixels = np.sum(building_mask)
    road_pixels = np.sum(road_mask)
    residential_building_pixels = np.sum(residential_building_mask)
    park_pixels = np.sum(park_mask)
    forest_pixels = np.sum(forest_mask)
    water_pixels = np.sum(water_mask)
    highway_pixels = np.sum(highway_mask)
    non_residential_building_pixels = np.sum(non_residential_building_mask)
    beach_pixels = np.sum(beach_mask)
    lake_pixels = np.sum(lake_mask)
    river_pixels = np.sum(river_mask)

    total_pixels = np.prod(building_mask.shape)
    building_percent = building_pixels / total_pixels
    urban_area_coverage = np.sum(elementwise_max(road_mask, residential_building_mask)) / total_pixels
    green_spaces_coverage = np.sum(elementwise_min(park_mask, forest_mask)) / total_pixels

    building_height_log = np.mean(elementwise_log(np.mean(elementwise_division(building_mask, 255))))
    urban_area_density = np.mean(elementwise_division(elementwise_sum(elementwise_product(road_mask, road_mask), road_mask), elementwise_product(elementwise_max(road_mask, highway_mask), road_mask)))
    urban_area_height = np.mean(elementwise_log(np.mean(elementwise_division(elementwise_max(road_mask, highway_mask), 255))))

    road_water_distance = min_pixel_distance_to_mask(elementwise_min(road_mask, water_mask))
    road_highway_distance = min_pixel_distance_to_mask(elementwise_min(road_mask, highway_mask))

    feature1 = np.mean(elementwise_log(elementwise_sum(elementwise_product(building_mask, building_mask), building_mask)))
    feature2 = np.mean(elementwise_product(elementwise_division(residential_building_mask, park_mask), park_mask))
    feature3 = np.mean(elementwise_exponentiate(elementwise_product(building_mask, water_mask), 2))
    feature4 = np.mean(elementwise_product(elementwise_division(road_mask, highway_mask), elementwise_division(road_mask, park_mask)))
    feature5 = np.mean(elementwise_product(elementwise_division(road_mask, highway_mask), elementwise_division(road_mask, beach_mask)))

    feature6 = np.mean(elementwise_product(elementwise_division(highway_mask, road_mask), elementwise_division(highway_mask, lake_mask)))
    feature7 = np.mean(elementwise_product(elementwise_division(residential_building_mask, highway_mask), elementwise_division(residential_building_mask, river_mask)))
    feature8 = np.mean(elementwise_product(elementwise_division(beach_mask, highway_mask), elementwise_division(beach_mask, lake_mask)))
    feature9 = np.mean(elementwise_product(elementwise_division(road_mask, highway_mask), elementwise_division(road_mask, park_mask)))
    feature10 = np.mean(elementwise_product(elementwise_division(road_mask, highway_mask), elementwise_division(road_mask, beach_mask)))
    feature11 = np.mean(elementwise_product(urban_area_density, urban_area_height))
    feature12 = np.mean(elementwise_product(elementwise_division(road_mask, highway_mask), elementwise_division(road_mask, water_mask)))
    feature13 = np.mean(elementwise_product(elementwise_division(road_mask, highway_mask), elementwise_division(road_mask, lake_mask)))
    feature14 = np.mean(elementwise_product(elementwise_division(beach_mask, highway_mask), elementwise_division(beach_mask, river_mask)))
    feature15 = np.mean(elementwise_product(elementwise_division(road_mask, highway_mask), elementwise_division(road_mask, park_mask)))

    return (building_pixels, road_pixels, residential_building_pixels, park_pixels, forest_pixels, water_pixels, highway_pixels, non_residential_building_pixels, beach_pixels, lake_pixels, river_pixels, 
            building_percent, urban_area_coverage, green_spaces_coverage, building_height_log, 
            feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15)
