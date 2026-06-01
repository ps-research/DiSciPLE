"""Verbatim primitive API specifications shown to the LLM.

These docstrings are transcribed exactly from the DiSciPLE supplementary
Appendix I. They are injected into the init / crossover / mutation prompts so
the LLM knows the available primitives. Per the paper's "no common sense"
ablation, the precise wording of these descriptions is load-bearing -- the LLM
relies on them to compose correct programs -- so they must not be paraphrased.

* ``POPULATION_API_SPEC``  -- Appendix I.1 (math/logic + ``segment`` only; no
  environment functions, no ``get_average``).
* ``POVERTY_AGB_API_SPEC`` -- I.1 plus the additional location/env primitives
  and ``get_average`` from Appendix I.2.
"""

# Appendix I.1 -- primitives for Population Density.
POPULATION_API_SPEC = '''def elementwise_max(matrix1, matrix2):
    """
    Compute the element-wise maximum of two matrices.

    Parameters:
    matrix1 (numpy.ndarray): First input matrix.
    matrix2 (numpy.ndarray): Second input matrix.

    Returns:
    numpy.ndarray: Element-wise maximum of the input matrices.
    """

def elementwise_min(matrix1, matrix2):
    """
    Compute the element-wise minimum of two matrices.

    Parameters:
    matrix1 (numpy.ndarray): First input matrix.
    matrix2 (numpy.ndarray): Second input matrix.

    Returns:
    numpy.ndarray: Element-wise minimum of the input matrices.
    """

def elementwise_sum(matrix1, matrix2):
    """
    Compute the element-wise sum of two matrices.

    Parameters:
    matrix1 (numpy.ndarray): First input matrix.
    matrix2 (numpy.ndarray): Second input matrix.

    Returns:
    numpy.ndarray: Element-wise sum of the input matrices.
    """

def elementwise_product(matrix1, matrix2):
    """
    Compute the element-wise product of two matrices.

    Parameters:
    matrix1 (numpy.ndarray): First input matrix.
    matrix2 (numpy.ndarray): Second input matrix.

    Returns:
    numpy.ndarray: Element-wise product of the input matrices.
    """

def elementwise_division(matrix1, matrix2):
    """
    Compute the element-wise division of two matrices.

    Parameters:
    matrix1 (numpy.ndarray): First input matrix.
    matrix2 (numpy.ndarray): Second input matrix.

    Returns:
    numpy.ndarray: Element-wise division of the input matrices.
    """

def matrix_scalar_multiplication(matrix, scalar):
    """
    Perform matrix scalar multiplication.

    Parameters:
    matrix (numpy.ndarray): Input matrix.
    scalar (int or float): Scalar value.

    Returns:
    numpy.ndarray: Result of matrix scalar multiplication.
    """

def elementwise_log(matrix):
    """
    Compute the element-wise logarithm of a matrix.

    Parameters:
    matrix (numpy.ndarray): Input matrix.

    Returns:
    numpy.ndarray: Element-wise logarithm of the input matrix.
    """

def elementwise_exponentiate(matrix, base):
    """
    Compute the exponentiation of each element of a matrix with a given base.

    Parameters:
    matrix (numpy.ndarray): Input matrix.
    base (int or float): Base value.

    Returns:
    numpy.ndarray: Exponentiated matrix.
    """

def min_pixel_distance_to_mask(mask):
    """
    Compute the minimum pixel distance from each pixel to a mask.

    Parameters:
    mask (numpy.ndarray): Binary mask array where 1 represents the mask and 0 represents the background.

    Returns:
    numpy.ndarray: Minimum pixel distance to the mask.
    """

def segment(im, text_prompt="trees"):
    """
    Segments a satellite image based on a text prompt. The text prompt can only take one concept at a time.

    Parameters:
    im (numpy.ndarray): An rgb satellite image.
    text_prompt (str): a text prompt

    Returns:
    mask (numpy.ndarray): Binary mask array where 1 represents the mask and 0 represents the background.
    Example:
    text_prompt can be but not limited to these:
    ["tennis", "skate park", "football field", "swimming pool", "cemetery", "multi-storey garage", "golf", "roundabout", "parking lot", "supermarket", "school", "marina", "baseball field", "fall", "pond", "airport", "beach", "bridge", "religious building", "residential building", "warehouse", "office building", "farmland", "university building", "forest", "lake", "nature reserve", "park", "sand", "soccer field", "equestrian club", "shooting range", "ice-rink", "commercial area", "garden", "dam", "railroad", "highway", "river", "wetland", "non-residential buildings", "coastline"]
    """
'''

# Appendix I.2 -- additional primitives for AGB and Poverty (used on top of I.1).
_POVERTY_AGB_EXTRA = '''def get_satellite_image(location):
    """
    Get the satellite image for a given location.
    Parameters:
    location (tuple): Tuple containing the latitude and longitude of the location.
    Returns:
    numpy.ndarray: Satellite image for the location.

    Can be used for segmentation ONLY.
    """

def get_temperature(location):
    """
    Get the average annual temperature for a given location.
    Parameters:
    location (tuple): Tuple containing the latitude and longitude of the location.
    Returns:
    float: Temperature for the location normalized between 0 and 255.
    """

def get_precipitation(location):
    """
    Get the average annual precipitation for a given location.
    Parameters:
    location (tuple): Tuple containing the latitude and longitude of the location.
    Returns:
    float: Precipitation for the location between 0 and 255.
    """

def get_elevation(location):
    """
    Get the elevation for a given location.
    Parameters:
    location (tuple): Tuple containing the latitude and longitude of the location.
    Returns:
    float: Digital Elevation for the location (scaled 0-8000) to 0-255.
    """

def get_nightlight_intensity(location):
    """
    Get the average annual nightlight intensity for a given location.
    Parameters:
    location (tuple): Tuple containing the latitude and longitude of the location.
    Returns:
    float: Nightlight intensity for the location (between 0 and 1).
    """

def get_average(segmented_image):
    """
    Get the average pixel value of a segmented image.

    Parameters:
    segmented_image (numpy.ndarray): Segmented image.

    Returns:
    float: Average pixel value of the segmented image.
    """
'''

# The poverty/AGB spec is I.1 followed by the I.2 additions ("uses all the
# above defined functions, plus the following").
POVERTY_AGB_API_SPEC = POPULATION_API_SPEC + "\n" + _POVERTY_AGB_EXTRA


def get_api_spec(benchmark_name: str) -> str:
    """Return the API-spec string the LLM should see for a given benchmark."""
    if benchmark_name == "population_density":
        return POPULATION_API_SPEC
    return POVERTY_AGB_API_SPEC
