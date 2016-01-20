import math


def calculate_distance_in_metres(lat1, lon1, lat2, lon2):

    r = 6371000 # Earth radius in metres

    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2) + (math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = r * c

    return d


"""def distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))

    try:
        arc = math.acos( cos )
    except:
        return 0

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc * 6371000 # Return as kilometers
"""

def strip(nodes, filter_distance):

    i=0


    for_removal = set()

    for x in nodes:
        for y in nodes:
            if x == y:
                continue

            # Get distance in meter

            distance = calculate_distance_in_metres(x.latitude, x.longitude, y.latitude, y.longitude)

            if distance < 10000:
                i += 1

    print(len(for_removal))
    print(i)



