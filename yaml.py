import json

# Simple yaml replacement using JSON for minimal use cases

def safe_load(stream):
    """Parse a stream into Python data structures."""
    return json.load(stream)

load = safe_load


def dump(data, stream=None):
    """Serialize ``data`` as JSON and write to ``stream`` if provided."""
    if stream is None:
        return json.dumps(data, indent=2)
    json.dump(data, stream, indent=2)
    return None

