from daemons.prefab import run
import sys

def daemonize(f, action, pidfile=None):
  if not pidfile:
    pidfile = f.__name__

  class NewDaemon(run.RunDaemon):
    def run(self):
      f()

  daemon = NewDaemon(pidfile="/tmp/{}.pid".format(pidfile))
  if (action == "start"):
    daemon.start()
  elif (action == "stop"):
    daemon.stop()
