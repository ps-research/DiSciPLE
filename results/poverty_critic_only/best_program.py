def estimator(location):
    satellite_image = get_satellite_image(location)
    temperature = get_temperature(location) / 255
    precipitation = get_precipitation(location) / 255
    elevation = get_elevation(location) / 8000
    nightlight_intensity = get_nightlight_intensity(location)

    avg_temp_log = np.mean(elementwise_log(temperature))
    avg_precip_log = np.mean(elementwise_log(precipitation))
    avg_elev_log = np.mean(elementwise_log(elevation))
    avg_night_log = np.mean(elementwise_log(nightlight_intensity))

    road_pixels = segment(satellite_image, "roads")
    road_pixels_count = np.sum(road_pixels)
    road_pixels_distance = np.mean(min_pixel_distance_to_mask(road_pixels))

    warehouse_pixels = segment(satellite_image, "warehouse")
    warehouse_pixels_count = np.sum(warehouse_pixels)
    warehouse_pixels_distance = np.mean(min_pixel_distance_to_mask(warehouse_pixels))

    university_building_pixels = segment(satellite_image, "university building")
    university_building_pixels_count = np.sum(university_building_pixels)

    park_pixels = segment(satellite_image, "park")
    park_pixels_count = np.sum(park_pixels)

    school_pixels = segment(satellite_image, "school")
    school_pixels_count = np.sum(school_pixels)

    wealth_index = np.mean([avg_temp_log, avg_precip_log, avg_elev_log, avg_night_log]) * (road_pixels_count + warehouse_pixels_count + university_building_pixels_count + school_pixels_count)

    feature1 = avg_temp_log
    feature2 = avg_precip_log
    feature3 = avg_elev_log
    feature4 = avg_night_log
    feature5 = road_pixels_count
    feature6 = road_pixels_distance
    feature7 = warehouse_pixels_count
    feature8 = warehouse_pixels_distance
    feature9 = university_building_pixels_count
    feature10 = park_pixels_count
    feature11 = school_pixels_count
    feature12 = wealth_index
    feature13 = np.mean(elementwise_log(np.mean(elementwise_product(avg_temp_log, avg_precip_log))))
    feature14 = np.mean(elementwise_log(np.mean(elementwise_product(avg_elev_log, avg_night_log))))
    feature15 = np.mean(elementwise_log(np.mean(elementwise_product(avg_temp_log, avg_elev_log))))
    feature16 = np.mean(elementwise_log(np.mean(elementwise_product(avg_precip_log, avg_night_log))))
    feature17 = np.mean(elementwise_log(np.mean(elementwise_product(road_pixels_count, warehouse_pixels_count))))
    feature18 = np.mean(elementwise_log(np.mean(elementwise_product(park_pixels_count, school_pixels_count))))

    return feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18
