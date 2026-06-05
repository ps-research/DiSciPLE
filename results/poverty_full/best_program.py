def estimator(location):
    satellite_image = get_satellite_image(location)
    nightlight = get_nightlight_intensity(location)
    nightlight_log = elementwise_log(np.clip(nightlight, 0, 1))
    school_mask = segment(satellite_image, 'school')
    warehouse_mask = segment(satellite_image, 'warehouse')
    road_mask = segment(satellite_image, 'roads')
    school_pixels = np.count_nonzero(school_mask)
    warehouse_pixels = np.count_nonzero(warehouse_mask)
    road_pixels = np.count_nonzero(road_mask)
    total_pixels = np.prod(satellite_image.shape)
    school_pixel_ratio = school_pixels / total_pixels
    warehouse_pixel_ratio = warehouse_pixels / total_pixels
    road_pixel_ratio = road_pixels / total_pixels
    temperature = get_temperature(location) / 255
    elevation = get_elevation(location) / 8000
    temperature_log = elementwise_log(temperature)
    nightlight_log_sum = elementwise_sum(elementwise_log(nightlight_log), elementwise_log(1 - nightlight_log))
    elevation_log = elementwise_log(elevation)
    feature2 = np.mean(elementwise_sum(temperature_log, nightlight_log_sum))
    feature3 = np.mean(elementwise_product(temperature_log, elevation_log))
    feature4 = np.mean(elementwise_sum(elementwise_log(school_pixel_ratio), elementwise_log(warehouse_pixel_ratio)))
    feature9 = np.mean(elementwise_product(elementwise_log(warehouse_pixel_ratio), elementwise_log(1 - school_pixel_ratio)))
    feature21 = np.mean(elementwise_sum(elementwise_log(warehouse_pixel_ratio), elementwise_log(road_pixel_ratio)))
    return (feature2, feature3, feature4, feature9, feature21)
