from shapefile_reader import get_shapefile_bounding_box

def main():
    shapefile_path = input("Enter the path to the shape file: ")
    try:
        bounding_box = get_shapefile_bounding_box(shapefile_path)
        print(f"Bounding Box: {bounding_box}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
