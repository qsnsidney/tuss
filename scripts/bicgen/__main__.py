from . import translation

# in_tipsy_file_path = './data/tipsy/LOW/LOW.bin'
in_tipsy_file_path = None
print('Info:', 'Loading from', in_tipsy_file_path)
translation.from_tipsy_into_bin(
    in_tipsy_file_path=in_tipsy_file_path, out_bin_file_path=None)
print('hello')
