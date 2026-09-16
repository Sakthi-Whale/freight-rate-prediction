import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat/2)**2 +
        np.cos(lat1) *
        np.cos(lat2) *
        np.sin(dlon/2)**2
    )

    return 2 * R * np.arcsin(np.sqrt(a))

def create_features(df):
    data = df.copy()

    data["geo_distance"] = haversine(
        data["pickup_lat"],
        data["pickup_lon"],
        data["delivery_lat"],
        data["delivery_lon"],
    )

    data["weight_per_mile"] = (
        data["weight"].abs() /
        (data["distance"] + 1)
    )

    data["market_distance"] = (
        data["market_index"] *
        data["distance"]
    )

    data["pickup_state"] = (
        data["pickup"].str[-2:]
    )

    data["delivery_state"] = (
        data["delivery"].str[-2:]
    )

    return data