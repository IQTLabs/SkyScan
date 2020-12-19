# Copyright (c) 2016 Johan Kanflo (github.com/kanflo)
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

import sys, os
import urllib2
import cStringIO
import json
import socket
import logging
import bing

log = logging.getLogger(__name__)

try:
    from PIL import Image
except ImportError:
    print("Pillow module not found, install using 'sudo pip install Pillow'")
    exit(1)


def getImage(searchTerm):
    searchTerm = "%s+logo" % (searchTerm)
    log.debug("Searching for %s" % searchTerm)

    try:
        imageUrls = bing.imageSearch(searchTerm, {"minWidth":400, "minHeight":400})
    except Exception as e:
        log.error("Bing exception error: %s" % (e))
        return None

    for url in imageUrls:
        log.debug(" Checking %s" % url)
        fileName, fileExtension = os.path.splitext(url)
        if fileExtension != ".svg" and fileExtension != ".gif":

            log.debug("Fetching %s" % url)
            opener = urllib2.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            try:
                file = cStringIO.StringIO(opener.open(url, timeout = 10).read())
                im = Image.open(file)
                return (im, url)
            except urllib2.HTTPError as e:
                log.error("HTTP error: %s for %s" % (e, url))
                log.error("searchTerm : '%s'" % (searchTerm))
                pass
            except UnicodeEncodeError as e:
                log.error("UnicodeEncodeError error: %s" % (e))
                log.error("searchTerm : '%s'" % (searchTerm))
                pass
            except urllib2.URLError as e:
                log.error("URLError error: %s for %s" % (e, url))
                log.error("searchTerm : '%s'" % (searchTerm))
                pass

    return (None, None)

def getProminentColor(searchTerm):
    (im, url) = getImage(searchTerm)
    if im == None:
        # In case we cannot find a color, make sure we don't end up here in 10 milliseconds
        return ((0,0,0), "error")

    histogram = {}
    limit = 10

    px = im.getpixel((0, 0))
    if isinstance(px, (int, long)):
        im = im.convert() # Conver to multi-layer image

    try:
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                px = im.getpixel((i,j))
                if px != (0, 0, 0) and px != (0, 0, 0, 0) and px != (255, 255, 255):
                    if abs(px[0]-px[1]) > limit or abs(px[0]-px[2]) > limit or abs(px[1]-px[2]) > limit:
                        if not px in histogram:
                            histogram[px] = 1
                        else:
                            histogram[px] = histogram[px] + 1
    except AttributeError as e:
        pass # Grayscale image?

    px_max = (0, 0, 0)
    max_count = 0
    for px in histogram:
        if histogram[px] > max_count:
            px_max = px
            max_count = histogram[px]

    return (((px_max[0], px_max[1], px_max[2])), url)

def loadColorData():
    global colors
    try:
        colors = json.load(open("logocolors.json"))
    except:
        colors = {}
    return colors

def getColor(key):
    global colors
    key = key.encode('utf-8', 'ignore')
    if key in colors:
        color = colors[key]["color"]
    else:
        (color, url) = getProminentColor(key)
        if color and url:
            colors[key] = {}
            colors[key]["color"] = color
            colors[key]["url"] = url
            with open("logocolors.json", "w+") as f:
                f.write(json.dumps(colors))
    return color
