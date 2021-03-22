FROM mikenye/piaware:latest

ADD run /etc/services.d/dump1090/

EXPOSE 80/tcp 30003/tcp 30005/tcp 30105/tcp 30978/tcp 30979/tcp

ENTRYPOINT [ "/init" ]

HEALTHCHECK --start-period=7200s --interval=600s CMD /scripts/healthcheck.sh