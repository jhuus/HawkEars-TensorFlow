# SQLite database interface for audio recordings.

import sqlite3
from types import SimpleNamespace

class Database:
    def __init__(self, filename='data/training.db'):
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

            # Record per data source, e.g. Xeno-Canto
            query = '''
                CREATE TABLE IF NOT EXISTS Source (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT NOT NULL)
            '''
            cursor.execute(query)

            # Record per top-level category, e.g. bird, frog, insect or machine
            query = '''
                CREATE TABLE IF NOT EXISTS Category (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT NOT NULL)
            '''
            cursor.execute(query)

            # Record per subcategory, which is species where relevant, but not for
            # "machine" sounds such as sirens and train whistles
            query = '''
                CREATE TABLE IF NOT EXISTS Subcategory (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    CategoryID INTEGER NOT NULL,
                    Name TEXT NOT NULL,
                    Synonym TEXT,
                    Code TEXT)
            '''
            cursor.execute(query)

            # Record per recording, including file name but not the recording itself.
            # Fields mostly come from Xeno-Canto, e.g.
            #   Recorder: person who submitted the recording
            #   License: e.g. "Creative-Commons"
            #   SoundType: e.g. "song" or "call"
            #   WasItSeen: was the bird seen?
            query = '''
                CREATE TABLE IF NOT EXISTS Recording (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    SourceID INTEGER NOT NULL,
                    SubcategoryID INTEGER NOT NULL,
                    FileName TEXT NOT NULL,
                    Seconds INTEGER)
            '''
            cursor.execute(query)

            # Record per spectrogram extracted from recordings, including the raw spectrogram data
            # and a link to the recording and offset within it
            query = '''
                CREATE TABLE IF NOT EXISTS Spectrogram (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    RecordingID INTEGER NOT NULL,
                    Value BLOB NOT NULL,
                    Offset REAL NOT NULL,
                    Audio BLOB,
                    Embedding BLOB)
            '''
            cursor.execute(query)

            # Create indexes for efficiency
            query = 'CREATE UNIQUE INDEX IF NOT EXISTS idx_subcategory_name ON Subcategory (Name)'
            cursor.execute(query)

            query = 'CREATE INDEX IF NOT EXISTS idx_recording_subcategoryid ON Recording (SubcategoryID)'
            cursor.execute(query)

            query = 'CREATE INDEX IF NOT EXISTS idx_spectrogram_recordingid ON Spectrogram (RecordingID)'
            cursor.execute(query)

            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database _create_tables: {e}')

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error as e:
            print(f'Error in database close: {e}')

# ------------------------------- #
# Source
# ------------------------------- #

    def insert_source(self, name):
        try:
            query = '''
                INSERT INTO Source (Name) Values (?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (name,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_source: {e}')

    def delete_source(self, field='ID', value=None):
        try:
            query = f'''
                DELETE FROM Source WHERE {field} = "{value}"
            '''

            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_source: {e}')

    def get_source(self, field=None, value=None):
        try:
            query = f'SELECT ID, Name FROM Source'
            if field is not None:
                query += f' WHERE {field} = "{value}"'

            query += f' Order BY ID'

            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, name = row
                result = SimpleNamespace(id=id, name=name)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_source: {e}')

# ------------------------------- #
# Category
# ------------------------------- #

    def insert_category(self, name):
        try:
            query = '''
                INSERT INTO Category (Name) Values (?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (name,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_category: {e}')

    def delete_category(self, field='ID', value=None):
        try:
            query = f'''
                DELETE FROM Category WHERE {field} = "{value}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_category: {e}')

    def get_category(self, field=None, value=None):
        try:
            query = f'SELECT ID, Name FROM Category'
            if field is not None:
                query += f' WHERE {field} = "{value}"'

            query += f' Order BY ID'

            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, name = row
                result = SimpleNamespace(id=id, name=name)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_category: {e}')


# ------------------------------- #
# Subcategory
# ------------------------------- #

    def insert_subcategory(self, category_id, name, synonym='', code=''):
        try:
            query = '''
                INSERT INTO Subcategory (CategoryID, Name, Synonym, Code) Values (?, ?, ?, ?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (category_id, name, synonym, code))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_subcategory: {e}')

    def delete_subcategory(self, field='ID', value=None):
        try:
            query = f'''
                DELETE FROM Subcategory WHERE {field} = "{value}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_subcategory: {e}')

    def get_subcategory(self, field=None, value=None):
        try:
            query = f'SELECT ID, CategoryID, Name, Synonym, Code FROM Subcategory'
            if field is not None:
                query += f' WHERE {field} = "{value}"'

            query += f' Order BY ID'

            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, categoryID, name, synonym, code = row
                result = SimpleNamespace(id=id, category_id=categoryID, name=name, code=code, synonym=synonym)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_subcategory: {e}')

    def get_subcategory_by_catid_and_subcat_name(self, category_id, name):
        try:
            query = f'''
                SELECT ID, CategoryID, Name, Synonym, Code FROM Subcategory
                WHERE CategoryID = "{category_id}" AND Name = "{name}"
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, categoryID, name, synonym, code = row
                result = SimpleNamespace(id=id, category_id=categoryID, name=name, code=code, synonym=synonym)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_subcategory_by_catid_and_subcat_name: {e}')

# ------------------------------- #
# Recording
# ------------------------------- #

    def insert_recording(self, source_id, subcategory_id, filename, seconds=0):
        try:
            query = '''
                INSERT INTO Recording (SourceID, SubcategoryID, FileName, Seconds)
                Values (?, ?, ?, ?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (source_id, subcategory_id, filename, seconds))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_recording: {e}')

    def delete_recording(self, field='ID', value=None):
        try:
            query = f'''
                DELETE FROM Recording WHERE {field} = "{value}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_recording: {e}')

    def delete_recording_by_subcat_name(self, subcategory_name):
        try:
            query = f'''
                DELETE FROM Recording WHERE SubcategoryID IN (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}")
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_recording_by_subcat_name: {e}')

    def get_recording(self, field=None, value=None):
        try:
            query = f'SELECT ID, SourceID, SubcategoryID, FileName, Seconds FROM Recording'
            if field is not None:
                query += f' WHERE {field} = "{value}"'

            query += f' Order BY ID'

            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, sourceID, subcategoryID, filename, seconds = row
                result = SimpleNamespace(id=id, source_id=sourceID, subcategory_id=subcategoryID, filename=filename, seconds=seconds)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_recording: {e}')

    def get_recording_by_subcat_name(self, subcategory_name):
        try:
            query = f'''
                SELECT ID, SourceID, SubcategoryID, FileName, Seconds
                FROM Recording
                WHERE SubcategoryID = (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}")
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, sourceID, subcategoryID, filename, seconds = row
                result = SimpleNamespace(id=id, source_id=sourceID, subcategory_id=subcategoryID, filename=filename, seconds=seconds)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_recording_by_subcategory_name: {e}')

    def get_recording_by_src_subcat_file(self, source_id, subcategory_id, filename):
        try:
            query = f'''
                SELECT ID, SourceID, SubcategoryID, FileName, Seconds
                FROM Recording
                WHERE SourceID = "{source_id}" AND SubcategoryID = "{subcategory_id}" AND FileName = "{filename}"
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, sourceID, subcategoryID, filename, seconds = row
                result = SimpleNamespace(id=id, source_id=sourceID, subcategory_id=subcategoryID, filename=filename, seconds=seconds)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_recording_by_src_subcat_file: {e}')

# ------------------------------- #
# Spectrogram
# ------------------------------- #

    def insert_spectrogram(self, recording_id, value, offset, embedding=None):
        try:
            query = 'INSERT INTO Spectrogram (RecordingID, Value, Offset, Embedding) Values (?, ?, ?, ?)'
            cursor = self.conn.cursor()
            cursor.execute(query, (recording_id, value, offset, embedding))
            cursor = self.conn.cursor()
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_spectrogram: {e}')

    def delete_spectrogram(self, field='ID', value=None):
        try:
            query = f'''
                DELETE FROM Spectrogram WHERE {field} = "{value}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_spectrogram: {e}')

    def delete_spectrogram_by_subcat_name(self, subcategory_name):
        try:
            query = f'''
                DELETE FROM Spectrogram WHERE RecordingID IN (SELECT ID From Recording WHERE SubcategoryID IN (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}"))
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_spectrogram_by_subcat_name: {e}')

    def get_spectrogram(self, field=None, value=None, include_audio=False, include_embedding=False):
        try:
            fields = 'ID, RecordingID, Value, Offset'
            if include_audio:
                fields += ', Audio'
            if include_embedding:
                fields += ', Embedding'

            query = f'SELECT {fields} FROM Spectrogram'
            if field is not None:
                query += f' WHERE {field} = "{value}"'

            query += f' Order BY ID'

            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None:
                return []

            results = []
            for row in rows:
                id, recordingID, value, offset = row[:4]
                if include_audio:
                    audio = row[4]
                    if include_embedding:
                        embedding = row[5]
                    else:
                        embedding = None
                elif include_embedding:
                    audio = None
                    embedding = row[4]
                else:
                    audio = None
                    embedding = None

                result = SimpleNamespace(id=id, recording_id=recordingID, value=value, offset=offset, audio=audio, embedding=embedding)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram: {e}')

    def get_spectrogram_by_recid_and_offset(self, recording_id, offset, include_audio=False, include_embedding=False):
        try:
            fields = 'ID, RecordingID, Value, Offset'
            if include_audio:
                fields += ', Audio'
            if include_embedding:
                fields += ', Embedding'

            query = f'''
                SELECT {fields} FROM Spectrogram
                WHERE RecordingID = {recording_id}
                AND Offset > {offset - .01}
                AND Offset < {offset + .01}
            '''

            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows is None or len(rows) == 0:
                return None

            row = rows[0]
            id, recordingID, value, offset = row[:4]
            if include_audio:
                audio = row[4]
                if include_embedding:
                    embedding = row[5]
                else:
                    embedding = None
            elif include_embedding:
                audio = None
                embedding = row[4]
            else:
                audio = None
                embedding = None

            result = SimpleNamespace(id=id, recording_id=recordingID, value=value, offset=offset, audio=audio, embedding=embedding)
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_by_recid_and_offset: {e}')

    def get_spectrogram_by_subcat_name(self, subcategory_name, include_audio=False, include_embedding=False):
        try:
            fields = 'Spectrogram.ID, RecordingID, Recording.FileName, Value, Offset'
            if include_audio:
                fields += ', Audio'
            if include_embedding:
                fields += ', Embedding'

            query = f'''
                SELECT {fields} FROM Spectrogram
                INNER JOIN Recording ON RecordingID = Recording.ID
                WHERE RecordingID IN
                    (SELECT ID FROM Recording WHERE SubcategoryID IN
                        (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}"))
                ORDER BY Recording.FileName, Offset
            '''

            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                id, recordingID, filename, value, offset = row[:5]
                if include_audio:
                    audio = row[5]
                    if include_embedding:
                        embedding = row[6]
                    else:
                        embedding = None
                elif include_embedding:
                    audio = None
                    embedding = row[5]
                else:
                    audio = None
                    embedding = None

                result = SimpleNamespace(id=id, recording_id=recordingID, filename=filename, value=value, offset=offset, audio=audio, embedding=embedding)
                results.append(result)

            return results

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_by_subcat_name: {e}')

    def get_spectrogram_count(self, subcategory_name):
        try:
            query = f'''
                SELECT COUNT(*)
                FROM Spectrogram
                WHERE RecordingID in
                    (SELECT ID From Recording WHERE SubcategoryID =
                        (SELECT ID FROM Subcategory WHERE NAME = "{subcategory_name}"))
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result != None:
                return result[0]
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_count: {e}')

    def update_spectrogram(self, id, field, value):
        try:
            query = f'''
                UPDATE Spectrogram SET {field} = ? WHERE ID = ?
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (value, id))

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database update_spectrogram: {e}')
