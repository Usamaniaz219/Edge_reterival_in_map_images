import cv2
import numpy as np

# Step 1: Dilate the Zoning Mask
def dilate_mask(mask, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

# Step 2: Find Contours in Zoning Mask
def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

# Step 3: Convert Contours to Polygons
def contours_to_polygons(contours, epsilon_factor=0.02):
    polygons = []
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(polygon)
    return polygons

# Step 4: Check Polygon Intersection
def polygon_intersection(zone_polygon, edge_polygons):
    intersecting_vertices = 0
    total_vertices = len(zone_polygon)
    
    for vertex in zone_polygon:
        point = tuple(vertex[0])  # Extract (x, y) coordinates from the polygon
        # print("point :",point)
        point = tuple([int(round(point[0]) ), int(round( point[1] )) ])
        for edge_polygon in edge_polygons:
            # print("edge_polygon type",type(edge_polygon))
            if cv2.pointPolygonTest(edge_polygon, point, False) >= 0:  # Check if point is inside edge polygon
                intersecting_vertices += 1
                break  # Move to the next vertex after finding one match

    # Check if 80% or more vertices intersect
    return (intersecting_vertices / total_vertices) >= 0.98

# Step 5: Retain or Discard Based on Intersection Percentage
def filter_polygons(zone_polygons, edge_polygons):
    retained_edges = []
    
    for zone_polygon in zone_polygons:
        if polygon_intersection(zone_polygon, edge_polygons):
            retained_edges.append(zone_polygon)
    
    return retained_edges

# Load Zoning Mask and Edge Image
zoning_mask = cv2.imread('/media/usama/6EDEC3CBDEC389B3/processed_Data_3_july_2024/denoised_data/ca_colma/ca_colma_4_mask.jpg', cv2.IMREAD_GRAYSCALE)
edge_mask = cv2.imread('contour_results_after_morph_gradients_19_sep_2024/data_12_Sep_5_gaussian_blur/ca_colma_output_mask_output_mask.jpg', cv2.IMREAD_GRAYSCALE)

# Process Zoning Mask
dilated_zone_mask = dilate_mask(zoning_mask, kernel_size=(3, 3), iterations=2)
zone_contours = find_contours(dilated_zone_mask)
# zone_contours = find_contours(zoning_mask)
zone_polygons = contours_to_polygons(zone_contours)

# Process Edge Mask
edge_contours = find_contours(edge_mask)
edge_polygons = contours_to_polygons(edge_contours)

# Find and Retain Intersecting Edges
retained_zone_polygons = filter_polygons(zone_polygons, edge_polygons)

# Visualization (optional)
output = np.zeros_like(zoning_mask)
cv2.drawContours(output, retained_zone_polygons, -1, (255, 255, 255), thickness=1)

# Save or display the result
cv2.imwrite('retained_edges.png', output)
