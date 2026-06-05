def estimator(location):
    img = get_satellite_image(location)
    mask = segment(img, "forest")
    avg_temperature = get_temperature(location) / 255
    avg_precipitation = get_precipitation(location) / 255
    elevation = get_elevation(location) / 8000
    nightlight_intensity = get_nightlight_intensity(location)
    
    avg_log_temp = elementwise_log(np.array([[avg_temperature]]))[0][0]
    avg_log_precip = elementwise_log(np.array([[avg_precipitation]]))[0][0]
    
    avg_product = elementwise_product(np.array([[get_average(mask)]]), np.array([[avg_log_temp]]))[0][0]
    avg_sum = elementwise_sum(np.array([[get_average(mask)]]), np.array([[avg_log_precip]]))[0][0]
    
    avg_min_dist = min_pixel_distance_to_mask(mask)
    avg_division = elementwise_division(np.array([[get_average(mask)]]), np.array([[avg_min_dist]]))[0][0]
    avg_exponentiate = elementwise_exponentiate(np.array([[get_average(mask)]]), 2)[0][0]
    
    feature1 = elementwise_product(img, mask)
    feature2 = elementwise_log(elementwise_division(img, mask))
    feature3 = elementwise_sum(img, mask)
    feature4 = min_pixel_distance_to_mask(mask)
    
    feature5 = elementwise_product(mask, elementwise_exponentiate(img, 2))
    feature6 = matrix_scalar_multiplication(img, 0.5)
    feature7 = elementwise_division(img, get_average(mask))
    
    feature8 = elementwise_product(elementwise_log(elementwise_max(img, mask)), elevation)
    feature9 = elementwise_product(elementwise_log(img), elevation)
    
    feature10 = elementwise_product(feature8, elementwise_log(feature8))
    feature11 = elementwise_product(feature8, feature7)
    
    feature12 = elementwise_product(feature10, feature6)
    feature13 = elementwise_product(feature10, feature11)
    feature14 = elementwise_product(feature10, elementwise_log(feature10))
    
    feature15 = elementwise_product(img, elementwise_log(img))
    feature16 = elementwise_sum(img, elementwise_log(img))
    feature17 = elementwise_division(feature15, feature16)
    feature18 = elementwise_product(feature15, feature7)
    feature19 = elementwise_product(feature15, feature11)
    
    feature20 = elementwise_product(img, mask)
    feature21 = elementwise_product(img, elementwise_log(img))
    feature22 = elementwise_sum(img, elementwise_log(img))
    feature23 = elementwise_division(img, get_average(mask))
    
    feature24 = elementwise_product(avg_log_temp, avg_log_precip)
    feature25 = elementwise_product(avg_product, avg_sum)
    feature26 = elementwise_product(avg_min_dist, avg_division)
    feature27 = elementwise_product(avg_exponentiate, feature7)
    
    return (
        feature1,
        feature2,
        feature3,
        feature4,
        feature5,
        feature6,
        feature7,
        feature8,
        feature9,
        feature10,
        feature11,
        feature12,
        feature13,
        feature14,
        feature15,
        feature16,
        feature17,
        feature18,
        feature19,
        feature20,
        feature21,
        feature22,
        feature23,
        feature24,
        feature25,
        feature26,
        feature27,
        elevation,
        nightlight_intensity,
        avg_log_temp,
        avg_log_precip,
        avg_product,
        avg_sum,
        avg_min_dist,
        avg_division,
        avg_exponentiate
    )
