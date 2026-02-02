python3 -c "import gi; gi.require_version('Gtk', '3.0'); from gi.repository import Gtk; d=Gtk.FileChooserDialog('test', None, Gtk.FileChooserAction.OPEN); d.run(); d.destroy()"
