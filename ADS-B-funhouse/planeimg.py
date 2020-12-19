# Copyright (c) 2015 Johan Kanflo (github.com/kanflo)
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys
import sqlite3
import datetime
try:
    from dateutil.parser import parse
except ImportError:
    print('sudo -H pip3 install python-dateutil')
    sys.exit(1)
import logging

log = logging.getLogger(__name__)

class Image(object):
  def __init__(self):
    self.icao24 = None
    self.image = None
    self.copyright = None

class ImageDB(object):
  def __init__(self, dbPath):
    global log
    self.path = dbPath
    self.db = sqlite3.connect(self.path)
    self.db.row_factory = sqlite3.Row
    self.db.text_factory = str
    c = self.db.cursor()
    c.execute("select count(*) from sqlite_master where type='table';")
    r = c.fetchone()
    if r[0] == 0:
      log.info("Initialized %s" % (self.path))
      self.dbInitialize()
    else:
      log.info("Opened %s" % (self.path))


  """Initialize database"""
  def dbInitialize(self):
    c = self.db.cursor()    
    c.execute("CREATE TABLE IF NOT EXISTS Images(Idx INTEGER PRIMARY KEY AUTOINCREMENT, ICAO24 varchar(6) NOT NULL, Image varchar(60), Copyright varchar(60));")
    self.db.commit()

  """Find aircraft image based on icao24"""
  def find(self, icao24):
    plane = None
    c = self.db.cursor()    
    c.execute("SELECT * FROM Images WHERE ICAO24=?;", (icao24,))
    row = c.fetchone()
    if row:
      plane = Image()
      plane.icao24 = row["ICAO24"]
      plane.image = row["Image"]
      plane.copyright = row["Copyright"]
    return plane

  """Add an image to the db"""
  def add(self, icao24, imageUrl, copyright = None):
    c = self.db.cursor()    
    c.execute("INSERT OR REPLACE INTO Images(ICAO24, Image, Copyright) VALUES(?, ?, ?);", (icao24, imageUrl, copyright))
    self.db.commit()

