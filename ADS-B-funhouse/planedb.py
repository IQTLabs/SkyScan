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

from typing import *
import requests as req
import json
import ast
import time
import os
import logging

hostname = "localhost"
port = 31541

# To keep the load down on the planedb server, we cache lookups and refresh
# every cache_max_age seconds. If entry has not been hit for that time we
# evict it from the cache to make sure we don't end up caching the world.
cache_max_age = 120

# Clean the cache every cache_clean_interval seconds
cache_clean_interval = 300

# Timestamp of last cache clean
last_cache_clean = time.time()

# Dicionaries keyed on "icao24" containing "fetched_ts", "hit_ts" and "data" (data may be 'False')
cache: Dict[str, Dict[str, str]] = {}

def _cache_clean():
    global last_cache_clean
    if time.time() - last_cache_clean > cache_clean_interval:
        global cache
        victims = []
        for what in cache:
            if time.time() - cache[what]["hit_ts"] > cache_max_age:
                victims.append(what)
        for what in victims:
            del cache[what]
        last_cache_clean = time.time()

def _cache_lookup(what: str) -> Union[Dict, None]:
    """Lookup 'what' in cache
       If found, return the data that may be 'False' if we have looked up
       'what' before but found nothing. Return None in case of cache misses

    Arguments:
        what {str} -- What to look for ;)

    Returns:
        Union[Dict, None] -- Dict describing what we found or None if data not in cache
    """
    _cache_clean()
    global cache
    if what in cache:
        t = time.time()
        item_age = t - cache[what]["fetched_ts"]
        if item_age < cache_max_age:
            cache[what]["hit_ts"] = time.time()
            return cache[what]["data"]
        else:
            del cache[what]
    return None 

def _cache_add(icao24: str, data: Dict):
    global cache
    cache[icao24] = {"fetched_ts" : time.time(), "hit_ts" : time.time(), "data" : data}

def init(_hostname: str, _port: int = 31541):
    global hostname
    global port
    hostname = _hostname
    port = _port

def lookup_aircraft(icao24: str):
    _cache = cache
    data = _cache_lookup(icao24)
    if data != None:
        return data
    try:
        resp = req.get("http://%s:%d/aircraft/%s" % (hostname, port, icao24))
        if resp.status_code == 200:
            data = ast.literal_eval(resp.text) # Works for single quotes
            _cache_add(icao24, data)
            return data
    except req.exceptions.ConnectionError:
        pass
    _cache_add(icao24, False)
    return False

def update_aircraft(icao24: str, data: Dict):
    url = "http://%s:%d/aircraft/%s" % (hostname, port, icao24)
    try:
        resp = req.post(url, data = data)
        if resp.status_code == 200:
            return resp.text == "OK"
    except req.exceptions.ConnectionError:
        pass
    return False

def lookup_airport(icao24: str):
    _cache = cache
    data = _cache_lookup(icao24)
    if data != None:
        return data
    try:
        resp = req.get("http://%s:%d/airport/%s" % (hostname, port, icao24))
        if resp.status_code == 200:
            _cache_add(icao24, data)
            return ast.literal_eval(resp.text) # Works for single quotes
    except req.exceptions.ConnectionError:
        pass
    _cache_add(icao24, False)
    return False

def update_airport(icao24: str, data: Dict):
    url = "http://%s:%d/airport/%s" % (hostname, port, icao24)
    try:
        resp = req.post(url, data = data)
        if resp.status_code == 200:
            return resp.text == "OK"
    except req.exceptions.ConnectionError:
        pass
    return False

def lookup_airline(icao24: str):
    _cache = cache
    data = _cache_lookup(icao24)
    if data != None:
        return data
    try:
        resp = req.get("http://%s:%d/airline/%s" % (hostname, port, icao24))
        print(resp.text)
        if resp.status_code == 200:
            _cache_add(icao24, data)
            return ast.literal_eval(resp.text) # Works for single quotes
    except req.exceptions.ConnectionError:
        pass
    _cache_add(icao24, False)
    return False

def update_airline(icao24: str, data: Dict):
    url = "http://%s:%d/airline/%s" % (hostname, port, icao24)
    try:
        resp = req.post(url, data = data)
        if resp.status_code == 200:
            return resp.text == "OK"
    except req.exceptions.ConnectionError:
        pass
    return False

def lookup_route(callsign: str):
    _cache = cache
    data = _cache_lookup(callsign)
    if data != None:
        return data
    try:
        resp = req.get("http://%s:%d/route/%s" % (hostname, port, callsign))
        if resp.status_code == 200:
            _cache_add(callsign, data)
            return ast.literal_eval(resp.text) # Works for single quotes
    except req.exceptions.ConnectionError:
        pass
    _cache_add(callsign, False)
    return False

def update_route(callsign: str, data: Dict):
    url = "http://%s:%d/route/%s" % (hostname, port, callsign)
    try:
        resp = req.post(url, data = data)
        if resp.status_code == 200:
            return resp.text == "OK"
        else:
            print(url)
            print(resp.status_code, resp.text)
    except req.exceptions.ConnectionError:
        pass
    return False

def add_imagecheck(icao24: str):
    url = "http://%s:%d/imagecheck/%s" % (hostname, port, icao24)
    try:
        resp = req.post(url)
        if resp.status_code == 200:
            return resp.text == "OK"
    except req.exceptions.ConnectionError:
        pass
    return False

def delete_imagecheck(icao24: str):
    url = "http://%s:%d/imagecheck/%s" % (hostname, port, icao24)
    try:
        resp = req.delete(url)
        if resp.status_code == 200:
            return resp.text == "OK"
    except req.exceptions.ConnectionError:
        pass
    return False

def get_imagechecks() -> List:
    url = "http://%s:%d/imagecheck" % (hostname, port)
    try:
        resp = req.get(url)
        if resp.status_code == 200:
            return json.loads(resp.text)
    except req.exceptions.ConnectionError:
        pass
    return []


# Use as a CLI tool
if __name__ == "__main__":
    import sys
    def dump(o):
        if o:
            for k in o:
                print("%20s : %s" % (k, o[k]))
        else:
            print("Not found")

    def google(icao24):
        import os, sys, subprocess
        plane = lookup_aircraft(sys.argv[2])
        if not "registration" in plane:
            print("No registration found for icao24 %s" % (icao24))
        else:
            url = "https://www.google.com/search?tbm=isch&q=%s" % plane["registration"]

            if sys.platform=='win32':
                print("win")
                os.startfile(url)
            elif sys.platform=='darwin':
                print("mac")
                subprocess.Popen(['open', url])
            else:
                print("linux?")
                try:
                    subprocess.Popen(['xdg-open', url])
                except OSError:
                    print("Don't know how to open a URL in this nachine, try %s" % url)

    # @todo: use switches, this is ugly as hell

    if 'PLANESERVER' in os.environ:
        server = os.environ['PLANESERVER']
    else:
        print("You need to set the environment variable PLANESERVER to point to your planeserver.")
        sys.exit(1)

    init(server)
    if len(sys.argv) < 2:
        print("Usage: %s [-l] [-g icao24] [-c icao24] [-d icao24] [-q icao24] [-r callsign] [-o airline] [-a airport] [-i icao24 [ -m <manufacturer> -t <type> -o <operator> -r <registration> -s <data source> -I <image url> ] ]" % sys.argv[0])
    else:
        if sys.argv[1] == '-o':
            dump(lookup_airline(sys.argv[2]))
        elif sys.argv[1] == '-a':
            dump(lookup_airport(sys.argv[2]))
        elif sys.argv[1] == '-r':
            dump(lookup_route(sys.argv[2]))
        elif sys.argv[1] == '-q':
            dump(lookup_aircraft(sys.argv[2]))
        elif sys.argv[1] == '-g':
            google(sys.argv[2])
        elif sys.argv[1] == '-c':
            add_imagecheck(sys.argv[2])
        elif sys.argv[1] == '-d':
            delete_imagecheck(sys.argv[2])
        elif sys.argv[1] == '-l':
            for plane in get_imagechecks():
                print("%8s https://www.google.com/search?tbm=isch&q=%s" % (plane["icao24"], plane["registration"]))

        elif sys.argv[1] == '-i':
            icao24 = sys.argv[2]
            man = None
            mdl = None
            reg = None
            op = None
            src = None
            img = None
            for i in range(3, len(sys.argv) - 1):
                if sys.argv[i] == '-m':
                    man = sys.argv[i+1]
                elif sys.argv[i] == '-t':
                    mdl = sys.argv[i+1]
                elif sys.argv[i] == '-o':
                    op = sys.argv[i+1]
                elif sys.argv[i] == '-r':
                    reg = sys.argv[i+1]
                elif sys.argv[i] == '-s':
                    src = sys.argv[i+1]
                elif sys.argv[i] == '-I':
                    img = sys.argv[i+1]

            plane = {'manufacturer' : man, 'model' : mdl, 'operator' : op, 'registration' : reg, 'image' : img, 'source' : src}
            print(plane)
            if (update_aircraft(icao24, plane)):
                print("ok")
            else:
                print("Update failed")
