import csv
from typing import List

class CSVReader:
    @staticmethod
    def read_csv(file_path: str) -> List[str]:
        paths = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                path = row[0].strip()
                if path not in paths:
                    paths.append(path)
        return paths
