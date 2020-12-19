# -*- coding: utf-8 -*-
#

import logging
import logging.handlers
import socket

"""
A module for centralized remote application logging
"""

def init(logger, host = "localhost", port = 5005, level = logging.INFO, appName = None, subSystem = None):
	remoteLogFormatter = logging.Formatter('%(asctime)s %(ip)s %(app)s %(subsys)s %(lineno)s %(name)s %(levelname)s %(message)s')

	remoteLogFilter = RemoteContextFilter()
	remoteLogFilter.appName = appName
	remoteLogFilter.subSystemName = subSystem
	hostName = socket.gethostname()
	if hostName == None or len(hostName) == 0:
		hostName = socket.gethostbyname(socket.gethostname())
	remoteLogFilter.hostName = hostName

	udpLogger = logging.handlers.DatagramHandler(host = host, port = port)
	udpLogger.setLevel(level)
	udpLogger.setFormatter(remoteLogFormatter)
	udpLogger.addFilter(remoteLogFilter)
	logger.addHandler(udpLogger)



class RemoteContextFilter(logging.Filter):
	appName = None
	subSystemName = None
	hostName = None

	def filter(self, record):
		record.app = self.appName
		record.subsys = self.subSystemName
		record.ip = self.hostName
		return True
