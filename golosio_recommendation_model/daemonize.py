from daemons.prefab import run
import sys

def daemonize(f, action):
  class NewDaemon(run.RunDaemon):
    def run(self):
      f()

  daemon = NewDaemon(pidfile="/tmp/{}.pid".format(f.__name__))
  if (action == "start"):
    daemon.start()
  elif (action == "stop"):
    daemon.stop()