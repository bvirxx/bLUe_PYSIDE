import os

from gi.repository import Colord
from gi.repository import Gio, Wnck
#from gi.repository import GLib

GIO_CANCELLABLE     = Gio.Cancellable.new()


def get_client_version():
   return # "%d.%d.%d" % (Colord.MAJOR_VERSION, Colord.MINOR_VERSION, Colord.MICRO_VERSION)


def default_profile_using_api(device_id):
   print ("===> Looking for default profile using API")
   print ("device_id = %s" % device_id)
   client = Colord.Client.new()
   client.connect_sync(GIO_CANCELLABLE)
   print ("colord daemon version:", client.get_daemon_version())
   print ('colord client version:', get_client_version())
   device = client.find_device_sync(device_id, GIO_CANCELLABLE)
   device.connect_sync(GIO_CANCELLABLE)
   print ("Connected to device id:", device.get_id())
   print ("  object path: %s"  % device.get_object_path())
   default_profile = device.get_default_profile()  # must be called BEFORE get_profiles !!!!
   profiles = device.get_profiles()
   print("get_profiles() found %d profiles:" % len(profiles))
   for profile in profiles:
      profile.connect_sync(GIO_CANCELLABLE)
      print ("  Profile id: %s" % profile.get_object_path())
      print ("  Filename:   %s" % profile.get_filename())

   print ("Default profile:", default_profile)
   default_profile.connect_sync(GIO_CANCELLABLE)
   print("  Profile id: %s" % default_profile.get_object_path())
   print("  Filename:   %s" % default_profile.get_filename())

def default_profile_using_colormgr(device_id):
   print ("===> Looking for default profile using colormgr")

   cmd = "colormgr --version"
   print ("Executing command: %s" % cmd)
   os.system(cmd)


   cmd = "colormgr find-device \"%s\"" % device_id
   print ("Executing command: %s" % cmd)
   os.system(cmd)


   cmd = "colormgr device-get-default-profile \"%s\"" % device_id
   print ("Executing command: %s" % cmd)
   os.system(cmd)





if __name__ == "__main__":
   a = Wnck.Screen.get_default()
   a.force_update()
   b=a.get_windows()[1].get_name()
   device_id = "xrandr-Virtual1"
   print("using colormgr")
   default_profile_using_colormgr( device_id )
   print("using api")
   default_profile_using_api( device_id )
