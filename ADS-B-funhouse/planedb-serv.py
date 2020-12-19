#!/usr/bin/env python3
#
# Copyright (c) 2019 Johan Kanflo (github.com/kanflo)
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

#
# A microservice for serving aircraft information
#

import sys
try:
    from flask import Flask, redirect, url_for, request, abort
    from peewee import *
    from playhouse.shortcuts import *
except ImportError:
    print("Missing requirements")
    print("sudo -H pip3 install flask peewee'")
    sys.exit(1)
import sys
import datetime
import logging
from logging.handlers import RotatingFileHandler
import datetime
import json

app = Flask(__name__)

if len(sys.argv) < 2:
    print("Usage: %s <database> [port]" % sys.argv[0])
    sys.exit(1)

db = SqliteDatabase(sys.argv[1])

if len(sys.argv) == 3:
    port = sys.argv[2]
else:
    port = 31541

class BaseModel(Model):
    class Meta:
        database = db # This model uses the "planes.db" database.

class Plane(BaseModel):
    icao24 = CharField(unique = True)
    added_on = DateTimeField(default = datetime.datetime.utcnow())
    updated_on = DateTimeField()
    manufacturer = CharField()
    model = CharField()
    operator = CharField()
    registration = CharField()
    source = CharField(null = True)
    is_military = BooleanField(default = False)
    image = CharField(null = True)

class Airline(BaseModel):
    icao = CharField(unique = True) #    3-letter ICAO code, if available.
    added_on = DateTimeField(default = datetime.datetime.utcnow())
    updated_on = DateTimeField()
    name = CharField()       #    Name of the airline.
    alias = CharField(null = True)      # Alias of the airline. For example, All Nippon Airways is commonly known as "ANA".
    iata = CharField()       # 2-letter IATA code, if available.
    callsign = CharField(null = True)   # Airline callsign.
    country = CharField()    # Country or territory where airline is incorporated.

class Airport(BaseModel):
    icao = CharField(unique = True)    # 4-letter ICAO code.
    added_on = DateTimeField(default = datetime.datetime.utcnow())
    updated_on = DateTimeField()
    name = CharField()    # Name of airport. May or may not contain the City name.
    city = CharField()    # Main city served by airport. May be spelled differently from Name.
    country = CharField() # Country or territory where airport is located. See countries.dat to cross-reference to ISO 3166-1 codes.
    iata = CharField()    # 3-letter IATA code. Null if not assigned/unknown.
    lat = DecimalField()  # Decimal degrees, usually to six significant digits. Negative is South, positive is North.
    lon = DecimalField()  # Decimal degrees, usually to six significant digits. Negative is West, positive is East.
    alt = DecimalField()  # In feet.

class Route(BaseModel):
    flight = CharField(unique = True)
    added_on = DateTimeField(default = datetime.datetime.utcnow())
    updated_on = DateTimeField()
    airline_icao = CharField()
    src_iata = CharField()
    dst_iata = CharField()

# Used as a TODO for planes that got weird images
class PlaneImageCheck(BaseModel):
    icao24 = CharField(unique = True)
    added_on = DateTimeField(default = datetime.datetime.utcnow())


db.connect()
db.create_tables([Plane, Airline, Airport, Route, PlaneImageCheck])

def get_aircraft(icao24):
    icao24 = icao24.lower()
    try:
        obj = Plane.get(icao24 = icao24)
    except Plane.DoesNotExist:
        abort(404)
    o = model_to_dict(obj)
    # model_to_dict does not format dates
    try:
        o['added_on'] = str(obj.added_on.strftime('%Y-%m-%d %H:%M:%S'))
    except AttributeError:
        # String is not a date string
        o['added_on'] = None
    try:
        o['updated_on'] = str(obj.updated_on.strftime('%Y-%m-%d %H:%M:%S'))
    except AttributeError:
        o['updated_on'] = None
    return o

def update_aircraft(icao24, post_dict):
    icao24 = icao24.lower()
    o = {}
    # Convert to ImmutableMultiDict to dict
    for k in post_dict:
        o[k] = post_dict[k]

    try:
        obj = Plane.get(icao24 = icao24)
        try:
            update_model_from_dict(obj, o)
        except AttributeError as e:
            print(e)
    except Plane.DoesNotExist:
        o['icao24'] = icao24
        obj = Plane(**o)

    obj.updated_on = datetime.datetime.utcnow()
    try:
        obj.save()
        return True
    except KeyError as e:
        logging.error("Exception occurred", exc_info=True)
        print(e)
        return False
    except IntegrityError as e:
        print("IntegrityError:")
        print(e)
        print(obj)
        return False
    except OperationalError as e:
        print("peewee operational error:")
        print(e)
        return False

def delete_aircraft(icao24):
    icao24 = icao24.lower()
    try:
        obj = Plane.get(icao24 = icao24)
        obj.delete_instance()
        return True
    except IndexError:
        return False

def get_imagechecks():
    from playhouse.shortcuts import model_to_dict, dict_to_model
    checks = []
    try:
        for check in PlaneImageCheck().select():
            plane = Plane.get(icao24 = check.icao24)
            checks.append(model_to_dict(plane))
    except PlaneImageCheck.DoesNotExist:
        return None
    return checks

def add_imagecheck(icao24):
    icao24 = icao24.lower()
    try:
        plane = Plane.get(icao24 = icao24)
    except Plane.DoesNotExist:
        return None
    obj = PlaneImageCheck()
    obj.icao24 = icao24
    obj.updated_on = datetime.datetime.utcnow()
    try:
        obj.save()
        return True
    except KeyError as e:
        logging.error("Exception occurred", exc_info=True)
        print(e)
        return False
    except IntegrityError as e:
        return False
    except OperationalError as e:
        print("peewee operational error:")
        print(e)
        return False

def delete_imagecheck(icao24):
    icao24 = icao24.lower()
    try:
        obj = PlaneImageCheck.get(icao24 = icao24)
        obj.delete_instance()
        return True
    except KeyError as e:
        logging.error("Exception occurred", exc_info=True)
        print(e)
        return False
    except IntegrityError as e:
        return False
    except PlaneImageCheck.DoesNotExist as e:
        return False
    except OperationalError as e:
        print("peewee operational error:")
        print(e)
        return False

def get_airport(ica_or_iata):
    ica_or_iata = ica_or_iata.upper()
    try:
        obj = Airport.get(icao = ica_or_iata)
    except Airport.DoesNotExist:
        try:
            obj = Airport.get(iata = ica_or_iata)
        except Airport.DoesNotExist:
            abort(404)
    o = model_to_dict(obj)
    # model_to_dict does not format dates
    o['lat'] = "%f" % obj.lat
    o['lon'] = "%f" % obj.lon
    o['alt'] = "%d" % obj.alt
    o['added_on'] = str(obj.added_on.strftime('%Y-%m-%d %H:%M:%S'))
    o['updated_on'] = str(obj.updated_on.strftime('%Y-%m-%d %H:%M:%S'))
    return o

def update_airport(icao, post_dict):
    icao = icao.upper()
    o = {}
    # Convert to ImmutableMultiDict to dict
    for k in post_dict:
        o[k] = post_dict[k]

    try:
        obj = Airport.get(icao = icao)
        try:
            update_model_from_dict(obj, o)
        except AttributeError as e:
            print(e)
    except Airport.DoesNotExist:
        o['icao'] = icao
        obj = Airport(**o)

    obj.updated_on = datetime.datetime.utcnow()
    try:
        obj.save()
        return True
    except KeyError as e:
        logging.error("Exception occurred", exc_info=True)
        print(e)
        return False
    except IntegrityError as e:
        print("IntegrityError:")
        print(e)
        print(obj)
        return False
    except OperationalError as e:
        print("peewee operational error:")
        print(e)
        return False

def delete_airport(icao):
    icao = icao.upper()
    try:
        obj = Airport.get(icao = icao)
        obj.delete_instance()
        return True
    except IndexError:
        return False


def get_airline(icao):
    icao = icao.upper()
    try:
        obj = Airline.get(icao = icao)
    except Airline.DoesNotExist:
        abort(404)
    o = model_to_dict(obj)
    # model_to_dict does not format dates
    try:
        o['added_on'] = str(obj.added_on.strftime('%Y-%m-%d %H:%M:%S'))
    except AttributeError:
        # String is not a date string
        o['added_on'] = None
    try:
        o['updated_on'] = str(obj.updated_on.strftime('%Y-%m-%d %H:%M:%S'))
    except AttributeError:
        o['updated_on'] = None
    return o

def update_airline(icao, post_dict):
    icao = icao.upper()
    o = {}
    # Convert to ImmutableMultiDict to dict
    for k in post_dict:
        o[k] = post_dict[k]

    try:
        obj = Airline.get(icao = icao)
        try:
            update_model_from_dict(obj, o)
        except AttributeError as e:
            print(e)
    except Airline.DoesNotExist:
        o['icao'] = icao
        obj = Airline(**o)

    obj.updated_on = datetime.datetime.utcnow()
    try:
        obj.save()
        return True
    except KeyError as e:
        logging.error("Exception occurred", exc_info=True)
        print(e)
        return False
    except IntegrityError as e:
        print("IntegrityError:")
        print(e)
        print(obj)
        return False
    except OperationalError as e:
        print("peewee operational error:")
        print(e)
        return False

def delete_airline(icao):
    icao = icao.upper()
    try:
        obj = Airline.get(icao = icao)
        obj.delete_instance()
        return True
    except IndexError:
        return False


def get_route(flight):
    flight = flight.upper()
    try:
        obj = Route.get(flight = flight)
    except Route.DoesNotExist:
        abort(404)
    o = model_to_dict(obj)
    # model_to_dict does not format dates
    try:
        o['added_on'] = str(obj.added_on.strftime('%Y-%m-%d %H:%M:%S'))
    except AttributeError:
        # String is not a date string
        o['added_on'] = None
    try:
        o['updated_on'] = str(obj.updated_on.strftime('%Y-%m-%d %H:%M:%S'))
    except AttributeError:
        o['updated_on'] = None
    return o

def update_route(flight, post_dict):
    flight = flight.upper()
    o = {}
    # Convert to ImmutableMultiDict to dict
    for k in post_dict:
        o[k] = post_dict[k]

    try:
        obj = Route.get(flight = flight)
        try:
            update_model_from_dict(obj, o)
        except AttributeError as e:
            print(e)
    except Route.DoesNotExist:
        o['flight'] = flight
        obj = Route(**o)

    obj.updated_on = datetime.datetime.utcnow()
    try:
        obj.save()
        return True
    except KeyError as e:
        logging.error("Exception occurred", exc_info=True)
        print(e)
        return False
    except IntegrityError as e:
        print("IntegrityError:")
        print(e)
        print(obj)
        return False
    except OperationalError as e:
        print("peewee operational error:")
        print(e)
        return False

def delete_route(flight):
    flight = flight.upper()
    try:
        obj = Route.get(flight = flight)
        obj.delete_instance()
        return True
    except IndexError:
        return False


@app.route("/")
def index():
    return "Index!"

@app.route("/aircraft")
def allIinfo():
    abort(404)

@app.route("/aircraft/<string:icao24>", methods = ['POST', 'GET', 'DELETE'])
def aircraft_info(icao24):
    icao24 = icao24.lower()
    if request.method == 'POST':
        logging.debug(request)
        logging.debug(request.form)
        if update_aircraft(icao24, request.form):
            return "OK"
        else:
            abort(400)
    elif request.method == 'GET':
        plane = get_aircraft(icao24)
        return "%s" % plane
    elif request.method == 'DELETE':
        if delete_aircraft(icao24):
            return "OK"
        else:
            abort(404)

# https://stackoverflow.com/questions/56554159/typeerror-object-of-type-datetime-is-not-json-serializable-with-serialize-fu
class DateTimeEncoder(json.JSONEncoder):
    def default(self, z):
        if isinstance(z, datetime.datetime):
            return (str(z))
        else:
            return super().default(z)

@app.route("/imagecheck", methods = ['GET'])
def imagechecks():
    checks = get_imagechecks()
    if checks:
        return json.dumps(checks, ensure_ascii=False, cls=DateTimeEncoder)
    else:
        abort(404)

@app.route("/imagecheck/<string:icao24>", methods = ['POST', 'DELETE'])
def imagecheck(icao24):
    icao24 = icao24.lower()
    if request.method == 'POST':
        if add_imagecheck(icao24):
            return "OK"
        else:
            abort(404)
    elif request.method == 'DELETE':
        if delete_imagecheck(icao24):
            return "OK"
        else:
            abort(404)

@app.route("/airport/<string:ica_or_iata>", methods = ['POST', 'GET', 'DELETE'])
def airport_info(ica_or_iata):
    ica_or_iata = ica_or_iata.upper()
    if request.method == 'POST':
        logging.debug(request)
        logging.debug(request.form)
        if update_airport(ica_or_iata, request.form):
            return "OK"
        else:
            abort(400)
    elif request.method == 'GET':
        plane = get_airport(ica_or_iata)
        return "%s" % plane
    elif request.method == 'DELETE':
        if delete_airport(ica_or_iata):
            return "OK"
        else:
            abort(404)

@app.route("/airline/<string:icao>", methods = ['POST', 'GET', 'DELETE'])
def airline_info(icao):
    icao = icao.upper()
    if request.method == 'POST':
        logging.debug(request)
        logging.debug(request.form)
        if update_airline(icao, request.form):
            return "OK"
        else:
            abort(400)
    elif request.method == 'GET':
        plane = get_airline(icao)
        return "%s" % plane
    elif request.method == 'DELETE':
        if delete_airline(icao):
            return "OK"
        else:
            abort(404)

@app.route("/route/<string:flight>", methods = ['POST', 'GET', 'DELETE'])
def route_info(flight):
    flight = flight.upper()
    if request.method == 'POST':
        logging.debug(request)
        logging.debug(request.form)
        if update_route(flight, request.form):
            return "OK"
        else:
            abort(400)
    elif request.method == 'GET':
        plane = get_route(flight)
        return "%s" % plane
    elif request.method == 'DELETE':
        if delete_route(flight):
            return "OK"
        else:
            abort(404)

@app.route("/image/<string:icao24>", methods = ['GET'])
def image(icao24):
    icao24 = icao24.lower()
    if request.method == 'GET':
        plane = get_aircraft(icao24)
        return "%s" % plane['image']
    else:
        abort(503)


if __name__ == "__main__":
#    level = logging.INFO # causes every HTTP op to be logged...
    level = logging.WARN
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    app.run(host = '0.0.0.0', port = port)
