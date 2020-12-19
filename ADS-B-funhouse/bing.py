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

import json
import os, urllib.request, re, urllib.parse

# Bing image search for python

headers = { 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}

# Search for 'keywords' and return image URLs in a list or None if, well, none
# are found or an error occurred
def imageSearch(keywords):
    filters = '+filterui:imagesize-large'
    current = 1
    url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(keywords) + '&first=' + str(current) + '&count=35&adlt=0&qft=' + filters
    request = urllib.request.Request(url, None, headers = headers)
    response = urllib.request.urlopen(request)
    html = response.read().decode('utf8')
    links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
    return links
