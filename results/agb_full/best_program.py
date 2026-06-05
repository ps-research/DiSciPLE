def estimator(location):
    img = get_satellite_image(location)
    mask = segment(img, 'forests') + segment(img, 'nature reserve') + segment(img, 'farmland') + segment(img, 'coastline') + segment(img, 'golf') + segment(img, 'non-residential buildings') + segment(img, 'school') + segment(img, 'park')
    temperature = get_temperature(location) / 255
    precipitation = get_precipitation(location) / 255
    elevation = get_elevation(location) / 8000
    avg_pixel = get_average(img)
    avg_pixel_masked = get_average(elementwise_min(mask, img))
    avg_pixel_distance_to_mask = get_average(min_pixel_distance_to_mask(mask))
    biomass = elementwise_product(elementwise_log(elementwise_max(mask, segment(img, 'beach'))), elementwise_product(temperature, elementwise_product(precipitation, elevation)))
    if avg_pixel > 140:
        biomass = elementwise_product(biomass, elementwise_exponentiate(avg_pixel, 0.5))
    if avg_pixel_masked > 130:
        biomass = elementwise_product(biomass, elementwise_exponentiate(avg_pixel_masked, 0.5))
    if avg_pixel_distance_to_mask > 120:
        biomass = elementwise_product(biomass, elementwise_exponentiate(avg_pixel_distance_to_mask, 0.5))
    for category in ['forests', 'farmland', 'golf', 'non-residential buildings', 'school', 'park']:
        if segment(img, category).any():
            biomass = elementwise_product(biomass, elementwise_exponentiate(avg_pixel, 1.02))
    return (temperature, elementwise_min(elementwise_product(temperature, precipitation), elementwise_product(elevation, avg_pixel)), avg_pixel, avg_pixel_distance_to_mask, elementwise_max(elementwise_product(biomass, elevation), elementwise_product(biomass, avg_pixel)))
