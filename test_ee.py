import ee
try:
    ee.Initialize()
    print("Earth Engine Initialization Successful!")
except Exception as e:
    print("Earth Engine Initialization Failed:")
    print(e)
