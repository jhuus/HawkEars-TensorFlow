# SQLite database interface for audio recordings.

import sqlite3

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
                    Code TEXT,
                    Ignore INTEGER,
                    Weight REAL)
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
                    URL Text NOT NULL,
                    FileName TEXT NOT NULL,
                    Seconds INTEGER,
                    Recorder TEXT,
                    License TEXT,
                    Quality TEXT,
                    Latitude TEXT,
                    Longitude TEXT,
                    SoundType TEXT,
                    Time TEXT,
                    Date TEXT,
                    Remark TEXT,
                    WasItSeen TEXT
                    )
            '''
            cursor.execute(query)

            # Record per spectrogram extracted from recordings, including the raw spectrogram data
            # and a link to the recording and offset within it
            query = '''
                CREATE TABLE IF NOT EXISTS Spectrogram (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    RecordingID INTEGER NOT NULL,
                    Value BLOB NOT NULL,
                    Encoding BLOB,
                    Offset REAL,
                    Type TEXT)
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

    def delete_recording(self, subcategory_name):
        try:
            query = f'''
                DELETE FROM Recording WHERE SubcategoryID IN (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}")
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_recording: {e}')

    def delete_recording_by_id(self, id):
        try:
            query = f'''
                DELETE FROM Recording WHERE ID = {id}
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_recording_by_id: {e}')

    def delete_spectrogram(self, subcategory_name):
        try:
            query = f'''
                DELETE FROM Spectrogram WHERE RecordingID IN (SELECT ID From Recording WHERE SubcategoryID IN (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}"))
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_spectrogram: {e}')

    def delete_spectrogram_by_id(self, id):
        try:
            query = f'''
                DELETE FROM Spectrogram WHERE ID = {id}
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_spectrogram_by_id: {e}')

    def delete_spectrogram_by_recording_id(self, recording_id):
        try:
            query = f'''
                DELETE FROM Spectrogram WHERE RecordingID = {recording_id}
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_spectrogram_by_recording_id: {e}')

    def delete_subcategory(self, subcategory_name):
        try:
            query = f'''
                DELETE FROM Subcategory WHERE Name = "{subcategory_name}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error in database delete_subcategory: {e}')

    def get_all_categories(self):
        try:
            query = f'''
                SELECT ID, Name FROM Category ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_all_categories: {e}')

    def get_category_by_id(self, id):
        try:
            query = f'''
                SELECT Name FROM Category WHERE ID = "{id}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result != None:
                return result[0]

            return result

        except sqlite3.Error as e:
            print(f'Error in database get_category_by_id: {e}')

    def get_category_by_name(self, name):
        try:
            query = f'''
                SELECT ID FROM Category WHERE Name = "{name}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result != None:
                return result[0]

            return result

        except sqlite3.Error as e:
            print(f'Error in database get_category_by_name: {e}')

    def get_num_spectrograms(self, subcategory_name):
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
            print(f'Error in database get_num_spectrograms: {e}')

    def get_all_recordings(self):
        try:
            query = f'''
                SELECT ID, SourceID, SubcategoryID, FileName, Seconds FROM Recording ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_all_recordings: {e}')

    def get_recording(self, id):
        try:
            query = f'''
                SELECT SourceID, SubcategoryID, URL, FileName, Seconds, Recorder, License,
                    Quality, Latitude, Longitude, SoundType, Time, Date, Remark, WasItSeen
                FROM Recording
                WHERE ID = "{id}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_recording: {e}')

    def get_recordings(self, source_id, subcategory_id):
        try:
            query = f'''
                SELECT ID, SourceID, SubcategoryID, URL, FileName, Seconds, Recorder, License,
                    Quality, Latitude, Longitude, SoundType, Time, Date, Remark, WasItSeen
                FROM Recording
                WHERE SourceID = "{source_id}" AND SubcategoryID = "{subcategory_id}"
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_recordings: {e}')

    def get_recordings_by_source(self, source_id):
        try:
            query = f'''
                SELECT ID, SourceID, SubcategoryID, URL, FileName, Seconds, Recorder, License,
                    Quality, Latitude, Longitude, SoundType, Time, Date, Remark, WasItSeen
                FROM Recording
                WHERE SourceID = "{source_id}"
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_recordings: {e}')

    def get_recordings_by_subcategory_name(self, subcategory_name):
        try:
            query = f'''
                SELECT ID, FileName, Seconds
                FROM Recording
                WHERE SubcategoryID = (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}")
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_recordings_by_subcategory_name: {e}')

    def get_recording_id(self, source_id, subcategory_id, filename):
        try:
            query = f'''
                SELECT ID
                FROM Recording
                WHERE SourceID = "{source_id}" AND SubcategoryID = "{subcategory_id}" AND FileName = "{filename}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result != None:
                return result[0]
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_recording_id: {e}')

    def get_recording_ids(self, subcategory_id):
        try:
            query = f'''
                SELECT ID
                FROM Recording
                WHERE SubcategoryID = "{subcategory_id}"
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_recording_ids: {e}')

    def get_all_spectrograms(self):
        try:
            query = f'''
                SELECT ID, RecordingID, Value, Offset FROM Spectrogram ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_all_spectrograms: {e}')

    def get_spectrogram(self, recording_id, offset):
        try:
            query = f'''
                SELECT ID, Value FROM Spectrogram
                WHERE RecordingID = {recording_id}
                AND Offset > {offset - .01}
                AND Offset < {offset + .01}
            '''

            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_id: {e}')

    def get_spectrogram_count_by_recid(self, recording_id):
        try:
            query = f'''
                SELECT COUNT(*) FROM Spectrogram
                WHERE RecordingID = {recording_id}
            '''

            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0]

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_count_by_recid: {e}')

    def get_spectrogram_count_by_subcat(self, subcategory_name):
        try:
            query = f'''
                SELECT COUNT(*) FROM Spectrogram
                WHERE RecordingID IN
                    (SELECT ID FROM Recording WHERE SubcategoryID IN
                        (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}"))
            '''

            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0]

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_count_by_subcat: {e}')

    def get_spectrograms_by_name(self, subcategory_name):
        try:
            query = f'''
                SELECT Value FROM Spectrogram
                WHERE RecordingID IN
                    (SELECT ID FROM Recording WHERE SubcategoryID IN
                        (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}"))
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrograms_by_name: {e}')

    def get_spectrogram_offsets_by_recording_id(self, recording_id):
        try:
            query = f'''
                SELECT Offset FROM Spectrogram
                WHERE RecordingID = {recording_id}
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrograms_by_recording_id: {e}')

    def get_spectrograms_by_recording_id(self, recording_id):
        try:
            query = f'''
                SELECT Value, Offset, Encoding FROM Spectrogram
                WHERE RecordingID = {recording_id}
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrograms_by_recording_id: {e}')

    def get_spectrograms_by_recording_id2(self, recording_id):
        try:
            query = f'''
                SELECT ID, Offset FROM Spectrogram
                WHERE RecordingID = {recording_id}
                ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrograms_by_recording_id2: {e}')

    def get_spectrogram_details_by_name(self, subcategory_name):
        try:
            query = f'''
                SELECT Recording.FileName, Offset, Value, Encoding FROM Spectrogram
                INNER JOIN Recording ON RecordingID = Recording.ID
                WHERE RecordingID IN
                    (SELECT ID FROM Recording WHERE SubcategoryID IN
                        (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}"))
                ORDER BY Recording.FileName, Offset
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_details_by_name: {e}')

    def get_spectrogram_details_by_name_2(self, subcategory_name):
        try:
            query = f'''
                SELECT Spectrogram.ID, Recording.ID, Value, Offset FROM Spectrogram
                INNER JOIN Recording ON RecordingID = Recording.ID
                WHERE RecordingID IN
                    (SELECT ID FROM Recording WHERE SubcategoryID IN
                        (SELECT ID FROM Subcategory WHERE Name = "{subcategory_name}"))
                ORDER BY Recording.FileName, Offset
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_details_by_name: {e}')

    def get_spectrogram_summaries(self, species_code):
        try:
            query = f'''
                SELECT Recording.Filename, Spectrogram.Offset FROM Spectrogram
                INNER JOIN Recording ON Spectrogram.RecordingID = Recording.ID
                WHERE RecordingID IN (SELECT ID From Recording WHERE SubcategoryID IN (SELECT ID FROM Subcategory WHERE Code = '{species_code}'))
                ORDER BY Recording.FileName, Spectrogram.Offset
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_spectrogram_summaries: {e}')

    def get_source_by_id(self, id):
        try:
            query = f'''
                SELECT Name FROM Source WHERE ID = "{id}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result != None:
                return result[0]

            return result

        except sqlite3.Error as e:
            print(f'Error in database get_source_by_name: {e}')

    def get_source_by_name(self, name):
        try:
            query = f'''
                SELECT ID FROM Source WHERE Name = "{name}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            if result != None:
                return result[0]

            return result

        except sqlite3.Error as e:
            print(f'Error in database get_source_by_name: {e}')

    def get_all_sources(self):
        try:
            query = f'''
                SELECT ID, Name FROM Source ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_source_by_name: {e}')

    def get_subcategory(self, category_id, name):
        try:
            query = f'''
                SELECT ID, Code FROM Subcategory WHERE CategoryID = "{category_id}" AND Name = "{name}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_subcategory: {e}')

    def get_subcategory_by_code(self, category_id, code):
        try:
            query = f'''
                SELECT ID, Name FROM Subcategory WHERE CategoryID = "{category_id}" AND Code = "{code}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_subcategory_by_code: {e}')

    def get_subcategory_by_name(self, name):
        try:
            query = f'''
                SELECT ID, Code FROM Subcategory WHERE Name = "{name}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_subcategory_by_name: {e}')

    def get_subcategory_details_by_name(self, name):
        try:
            query = f'''
                SELECT ID, CategoryID, Code, Synonym, Ignore, Weight FROM Subcategory WHERE Name = "{name}"
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_subcategory_by_name: {e}')

    def get_all_subcategories(self):
        try:
            query = f'''
                SELECT ID, Name, Code, Synonym, CategoryID FROM Subcategory ORDER BY ID
            '''
            cursor = self.conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result

        except sqlite3.Error as e:
            print(f'Error in database get_all_subcategories: {e}')

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

    def insert_recording(self, source_id, subcategory_id, url, filename, seconds = 0, recorder = "", license = "",
                         quality = "", latitude = "", longitude = "", sound_type = "", time = "", date = "", remark = "",
                         was_it_seen = ""):
        try:
            query = '''
                INSERT INTO Recording (SourceID, SubcategoryID, URL, FileName, Seconds, Recorder, License, Quality,
                                       Latitude, Longitude, SoundType, Time, Date, Remark, WasItSeen)
                Values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (source_id, subcategory_id, url, filename, seconds, recorder, license, quality,
                                   latitude, longitude, sound_type, time, date, remark, was_it_seen))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_recording: {e}')

    def insert_spectrogram(self, recording_id, value, offset, encoding=None, type=''):
        try:
            if encoding is None:
                query = '''
                    INSERT INTO Spectrogram (RecordingID, Value, Offset, Type)
                    Values (?, ?, ?, ?)
                '''
                cursor = self.conn.cursor()
                cursor.execute(query, (recording_id, value, offset, type))
            else:
                query = '''
                    INSERT INTO Spectrogram (RecordingID, Value, Offset, Type, Encoding)
                    Values (?, ?, ?, ?, ?)
                '''
                cursor = self.conn.cursor()
                cursor.execute(query, (recording_id, value, offset, type, encoding))

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_spectrogram: {e}')

    def insert_source(self, source):
        try:
            query = '''
                INSERT INTO Source (Name) Values (?)
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (source,))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database insert_source: {e}')

    def insert_subcategory(self, category_id, name, synonym='', code = ''):
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

    def update_recording_source_id(self, id, new_source_id):
        try:
            query = '''
                UPDATE Recording SET SourceID = ? WHERE ID = ?
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (new_source_id, id))

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database update_recording_source_id: {e}')

    def update_spectrogram_encoding(self, id, encoding):
        try:
            query = '''
                UPDATE Spectrogram SET Encoding = ? WHERE ID = ?
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (encoding, id))

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database update_spectrogram_encoding: {e}')

    def update_spectrogram_recording_id(self, id, new_recording_id):
        try:
            query = '''
                UPDATE Spectrogram SET RecordingID = ? WHERE ID = ?
            '''
            cursor = self.conn.cursor()
            cursor.execute(query, (new_recording_id, id))

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f'Error in database update_spectrogram_recording_id: {e}')

