from shape_file_reader import ShapeFileReader

def main():
    file_path = "path/to/shapefile.shp"
    reader = ShapeFileReader(file_path)
    bounding_box = reader.get_bounding_box()
    print(f"Bounding Box: {bounding_box}")

if __name__ == "__main__":
    main()
