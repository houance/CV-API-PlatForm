import configparser


class configReader:
    def __init__(self, path: str):
        self.reader = configparser.ConfigParser()
        self.reader.read(path)

    def readValues(self):
        values = []
        sections = self.reader.sections()
        for section in sections:
            tempValues = []
            items = self.reader.items(section)
            for item in items:
                tempValues.append(item[1])
            values.append(tempValues)
        return values
