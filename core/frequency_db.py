# SQLite database interface for eBird barchart data.

import sqlite3
from types import SimpleNamespace
import zlib

import numpy as np

class Frequency_DB:
    def __init__(self, filename='data/frequency.db'):
        self.conn = None
        try:
            self.conn = sqlite3.connect(filename)
            self._create_tables()
        except sqlite3.Error as e:
            print(f'Error in database init: {e}')

    # create tables if they don't exist
    def _create_tables(self):
        try:
            cursor = self.conn.cursor()

            query = '''
                CREATE TABLE IF NOT EXISTS County (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT NOT NULL,
                    Code TEXT NOT NULL,
                    MinX REAL NOT NULL,
                    MaxX REAL NOT NULL,
                    MinY REAL NOT NULL,
                    MaxY REAL NOT NULL
                    )
            '''
            cursor.execute(query)

            query = '''
                CREATE TABLE IF NOT EXISTS Species (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT NOT NULL
                    )
            '''
            cursor.execute(query)

            query = '''
                CREATE TABLE IF NOT EXISTS Frequency (
                    CountyID INTEGER NOT NULL,
                    SpeciesID INTEGER NOT NULL,
                    Value BLOB NOT NULL
                    )
            '''
            cursor.execute(query)

            # Create indexes for efficiency
            query = 'CREATE UNIQUE INDEX IF NOT EXISTS idx_species_name ON Species (Name)'
            cursor.execute(query)

            query = 'CREATE INDEX IF NOT EXISTS idx_county_id ON Frequency (CountyID)'
            cursor.execute(query)

            query = 'CREATE INDEX IF NOT EXISTS idx_species_id ON Frequency (SpeciesID)'
            cursor.execute(query)

            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database _create_tables: {e}')

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error as e:
            print(f'Error in database close: {e}')

    def get_all_counties(self):
        try:
            query = f'''
                SELECT ID, Name, Code, MinX, MaxX, MinY, MaxY FROM County ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                id, name, code, min_x, max_x, min_y, max_y = row
                county = SimpleNamespace(id=id, name=name, code=code, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
                results.append(county)

            return results
        except sqlite3.Error as e:
            print(f'Error in database get_all_counties: {e}')

    def get_all_species(self):
        try:
            query = f'''
                SELECT ID, Name FROM Species ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                id, name = row
                species = SimpleNamespace(id=id, name=name)
                results.append(species)

            return results
        except sqlite3.Error as e:
            print(f'Error in database get_all_species: {e}')

    def get_frequencies(self, county_id, species_name):
        try:
            query = f'''
                SELECT SpeciesID, Value FROM Frequency WHERE CountyID = {county_id}
                    AND SpeciesID = (SELECT ID From Species WHERE Name = "{species_name}")
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()

            if result is None:
                return []

            species_id, compressed = result
            bytes = zlib.decompress(compressed)
            values = np.frombuffer(bytes, dtype=np.uint8) / (4 * 255)
            values = values.astype(np.float32)

            results = []
            for i, value in enumerate(values):
                result = SimpleNamespace(county_id=county_id, species_id=species_id, week_num=i + 1, value=value)
                results.append(result)

            return results
        except sqlite3.Error as e:
            print(f'Error in database get_frequencies_by_county_id: {e}')

    def insert_county(self, name, code, min_x, max_x, min_y, max_y):
        try:
            query = '''
                INSERT INTO County (Name, Code, MinX, MaxX, MinY, MaxY) Values (?, ?, ?, ?, ?, ?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (name, code, min_x, max_x, min_y, max_y))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_county: {e}')

    def insert_frequencies(self, county_id, species_id, value):
        try:
            # value is a numpy array of 48 floats (a frequency per week, four weeks/month);
            # convert it to a byte array and zip that to keep the database small
            value = np.minimum(value, 0.25) * 4.0 # increase granularity at the cost of losing values > 0.25
            value = value * 255
            np_bytes = value.astype(np.uint8)
            bytes = np_bytes.tobytes()
            compressed = zlib.compress(bytes)

            query = '''
                INSERT INTO Frequency (CountyID, SpeciesID, Value) Values (?, ?, ?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (county_id, species_id, compressed))
            self.conn.commit()

        except sqlite3.Error as e:
            print(f'Error in database insert_frequencies: {e}')

    def insert_species(self, name):
        try:
            query = '''
                INSERT INTO Species (Name) Values (?)
            '''

            cursor = self.conn.cursor()
            cursor.execute(query, (name,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_species: {e}')
