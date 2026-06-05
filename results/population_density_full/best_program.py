def estimator(image):
    roads_mask = segment(image, 'roads')
    residential_buildings_mask = segment(image, 'residential buildings')
    commercial_area_mask = segment(image, 'commercial area')
    non_residential_buildings_mask = elementwise_max(residential_buildings_mask, commercial_area_mask)
    non_residential_feature = np.mean(elementwise_sum(non_residential_buildings_mask, non_residential_buildings_mask))
    urban_area_pixels = np.sum(elementwise_max(roads_mask, residential_buildings_mask))
    green_spaces_mask = segment(image, 'forest') + segment(image, 'park')
    green_spaces_mask = elementwise_min(green_spaces_mask, green_spaces_mask)
    park_pixels = np.sum(elementwise_product(green_spaces_mask, green_spaces_mask))
    park_density_feature = park_pixels / np.sum(green_spaces_mask) if np.sum(green_spaces_mask) > 0 else 0
    coastal_structures_mask = elementwise_max(segment(image, 'coastline'), segment(image, 'beach'))
    coastal_structures_pixels = np.sum(coastal_structures_mask)
    coastal_structures_feature = coastal_structures_pixels / np.sum(coastal_structures_mask) if np.sum(coastal_structures_mask) > 0 else 0
    return (np.sqrt(urban_area_pixels), park_density_feature, non_residential_feature, coastal_structures_feature)
