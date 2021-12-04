try:
    import yt
except Exception as e:
    print('Try `pip3 install yt`')
from .. import core

# https://yt-project.org/doc/cookbook/tipsy_notebook.html


def from_tipsy_into_bin(in_tipsy_file_path, out_bin_file_path):
    system_state = deserialize_system_state_from_tipsy(in_tipsy_file_path)
    core.serde.serialize_system_state_into_bin(system_state, out_bin_file_path)


def deserialize_system_state_from_tipsy(tipsy_file_path):
    pass
