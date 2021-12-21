from datetime import datetime


def get_pf_timestamp_str():
    '''
    Get a pathname friendly timestamp string.
    Down to seconds
    '''
    return datetime.now().strftime("%Y%m%d_%H%M%S")
