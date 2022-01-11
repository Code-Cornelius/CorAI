import json, base64
import zlib

ZIPJSON_KEY = 'base64(zip(o))'
COMPRESSION_LEVEL = 9


def zip_json(json_to_zip):
    """
    Zip a json serializable object.
    Args:
        json_to_zip: a Json serializable object.

    Returns:
        A dictionary with:
         - a key describing the encoding settings,
         - a value representing the zlib compressed json.
    """

    return {
        ZIPJSON_KEY: base64.b64encode(
            zlib.compress(
                json.dumps(json_to_zip).encode('utf-8'),
                COMPRESSION_LEVEL
            )
        ).decode('ascii')
    }


def unzip_json(json_to_unzip):
    """
    Unzip a json.

    Args:
        json_to_unzip: Must be json of the format:
            - a key corresponding to ZIPJSON_KEY,
            - a value corresponding to the zlib compressed json.

    Returns:
        The decompressed json.
    """
    # check if the json was encoded in the right way for our decoder
    assert (json_to_unzip[ZIPJSON_KEY])
    assert (set(json_to_unzip.keys()) == {ZIPJSON_KEY})

    try:
        json_to_unzip = zlib.decompress(base64.b64decode(json_to_unzip[ZIPJSON_KEY]))
    except:
        raise RuntimeError("Could not decode/unzip the contents")

    try:
        json_to_unzip = json.loads(json_to_unzip)
    except:
        raise RuntimeError("Could interpret the unzipped contents")

    return json_to_unzip


def is_jsonable(x):
    # reference
    # https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python
    try:
        json.dumps(x)
        return True
    except:
        return False