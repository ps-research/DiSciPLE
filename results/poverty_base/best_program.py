def estimator(location):
    img = get_satellite_image(location)
    mask = segment(img, "residential building")
    avg = get_average(mask)
    temp = get_temperature(location) / 255
    precip = get_precipitation(location) / 255
    elev = get_elevation(location) / 8000
    nightlight = get_nightlight_intensity(location)
    road_mask = segment(img, "roads")
    road_pixels = np.where(road_mask == 1)[0]
    road_pixels_count = len(road_pixels)
    road_pixels_distance = np.mean(min_pixel_distance_to_mask(road_mask))
    water_mask = segment(img, "water")
    water_pixels = np.where(water_mask == 1)[0]
    water_pixels_count = len(water_pixels)
    water_pixels_distance = np.mean(min_pixel_distance_to_mask(water_mask))
    greenery_mask = segment(img, "forest")
    greenery_pixels = np.where(greenery_mask == 1)[0]
    greenery_pixels_count = len(greenery_pixels)
    greenery_pixels_distance = np.mean(min_pixel_distance_to_mask(greenery_mask))
    poverty_index = np.mean(elementwise_log(np.array([avg, 1 - avg]))) * 0.3 + np.mean(elementwise_log(np.array([temp, 1 - temp]))) * 0.15 + np.mean(elementwise_log(np.array([precip, 1 - precip]))) * 0.15 + np.mean(elementwise_log(np.array([elev, 1 - elev]))) * 0.1 + road_pixels_count * 0.05 + road_pixels_distance * 0.05
    wealth_index = poverty_index + avg * 0.1 + road_pixels_count * 0.1 + road_pixels_distance * 0.1 + np.mean(elementwise_log(np.array([temp, 1 - temp]))) * 0.08 + np.mean(elementwise_log(np.array([precip, 1 - precip]))) * 0.08 + np.mean(elementwise_log(np.array([elev, 1 - elev]))) * 0.08 + np.mean(elementwise_log(np.array([nightlight, 1 - nightlight]))) * 0.08
    water_index = np.mean(elementwise_log(np.array([water_pixels_count, 1 - water_pixels_count]))) * 0.1 + water_pixels_distance * 0.05
    greenery_index = np.mean(elementwise_log(np.array([greenery_pixels_count, 1 - greenery_pixels_count]))) * 0.1 + greenery_pixels_distance * 0.05
    return (poverty_index, 
            np.mean(elementwise_log(np.array([avg, 1 - avg]))), 
            np.mean(elementwise_log(np.array([temp, 1 - temp]))), 
            np.mean(elementwise_log(np.array([precip, 1 - precip]))), 
            np.mean(elementwise_log(np.array([elev, 1 - elev]))), 
            np.mean(elementwise_log(np.array([nightlight, 1 - nightlight]))), 
            road_pixels_count, 
            road_pixels_distance, 
            avg * 0.05, 
            temp * 0.05, 
            precip * 0.05, 
            elev * 0.05, 
            nightlight * 0.05, 
            water_index, 
            greenery_index, 
            wealth_index)
