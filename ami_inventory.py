import collections
import csv
import datetime
import hashlib
import logging
import os
import pathlib
import re
import time
import urllib.parse
from sys import platform as _sys_platform
from typing import Literal, Optional, Union, Generator

import dateutil.parser
from dateutil.parser import ParserError

logger = logging.getLogger(__name__)

EXIF_DATETIME_STR_FORMAT = "%Y:%m:%d %H:%M:%S"
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg")
IMAGE_BASE_URL = "https://object-arbutus.cloud.computecanada.ca/ami-trapdata/"
FilePath = Union[pathlib.Path, str]

# Use placeholder for unimported PIL


def absolute_path(
    path: str, base_path: Union[pathlib.Path, str, None]
) -> Union[pathlib.Path, None]:
    """
    Turn a relative path into an absolute path.
    """
    if not path:
        return None
    elif not base_path:
        return pathlib.Path(path)
    elif str(base_path) in path:
        return pathlib.Path(path)
    else:
        return pathlib.Path(base_path) / path


def archive_file(filepath):
    """
    Rename an existing file to `<filepath>/<filename>.bak.<timestamp>`
    """
    filepath = pathlib.Path(filepath)
    if filepath.exists():
        suffix = f".{filepath.suffix}.backup.{str(int(time.time()))}"
        backup_filepath = filepath.with_suffix(suffix)
        logger.info(f"Moving existing file to {backup_filepath}")
        filepath.rename(backup_filepath)
        return backup_filepath


def find_timestamped_folders(path):
    """
    Find all directories in a given path that have
    dates / timestamps in the name.

    This should be the nightly folders from the trap data.

    >>> pathlib.Path("./tmp/2022_05_14").mkdir(exist_ok=True, parents=True)
    >>> pathlib.Path("./tmp/nope").mkdir(exist_ok=True, parents=True)
    >>> find_timestamped_folders("./tmp")
    OrderedDict([(datetime.datetime(2022, 5, 14, 0, 0), PosixPath('tmp/2022_05_14'))])
    """
    logger.debug("Looking for nightly timestamped folders")
    nights = collections.OrderedDict()

    def _preprocess(name):
        return name.replace("_", "-")

    dirs = sorted(pathlib.Path(path).iterdir())
    for d in dirs:
        # @TODO use yield?
        try:
            date = dateutil.parser.parse(_preprocess(d.name))
        except Exception:
            # except dateutil.parser.ParserError:
            pass
        else:
            logger.debug(f"Found nightly folder for {date}: {d}")
            nights[date] = d

    return nights


def import_pil():
    """
    Import the Pillow library and return it.
    """
    global PIL
    if PIL is None:
        try:
            import PIL
            import PIL.ExifTags
            import PIL.Image
        except ModuleNotFoundError:
            logger.error(
                "Cannot read EXIF tags. Please install it with 'pip install pillow'"
            )
            raise
    return PIL


def get_exif(img_path):
    """
    Read the EXIF tags in an image file
    """
    import_pil()

    img = PIL.Image.open(img_path)
    img_exif = img.getexif()
    tags = {}

    for key, val in img_exif.items():
        if key in PIL.ExifTags.TAGS:
            name = PIL.ExifTags.TAGS[key]
            # logger.debug(f"EXIF tag found'{name}': '{val}'")
            tags[name] = val

    return tags


def construct_exif(
    timestamp: Optional[datetime.datetime] = None,
    description: Optional[str] = None,
    location: Optional[dict] = None,
    other_tags: Optional[dict] = None,
    # existing_exif: Optional[PIL.Image.Exif] = None,) -> PIL.Image.Exif:
    existing_exif: Optional[dict] = None,
) -> dict:
    """
    Construct an EXIF class using human readable keys.
    Can be save to a Pillow image using:

    >>> image = PIL.Image.open("./trapdata/tests/images/denmark/20220811005907-00-78.jpg")
    >>> existing_exif = image.getexif()
    >>> exif_data = construct_exif(description="hi!", existing_exif=existing_exif)
    >>> image.save("test_with_exif.jpg", exif=exif_data)
    """

    exif = existing_exif or PIL.Image.Exif()

    name_to_code = {name: code for code, name in PIL.ExifTags.TAGS.items()}

    if timestamp:
        timestamp_str = timestamp.strftime(EXIF_DATETIME_STR_FORMAT)
        exif[name_to_code["DateTime"]] = timestamp_str
        exif[name_to_code["DateTimeOriginal"]] = timestamp_str
        exif[name_to_code["DateTimeDigitized"]] = timestamp_str

    if description:
        exif[name_to_code["ImageDescription"]] = description
        exif[name_to_code["UserComment"]] = description

    if location:
        raise NotImplementedError

    if other_tags:
        for name in other_tags:
            if name not in name_to_code:
                raise Exception(f"Unknown EXIF tag '{name}'")
            else:
                code = name_to_code[name]
                value = other_tags[name]
                logger.debug(f"Adding EXIF tag {name} ({code}) with value {value}")
                exif[code] = value

    return exif


def get_image_dimensions(img_path) -> Optional[tuple[int, int]]:
    """
    Get the dimensions of an image without loading it into memory.
    """
    try:
        import imagesize
    except ModuleNotFoundError:
        logger.error(
            "Could not calculate image size. Please install 'imagesize' with 'pip install imagesize'"
        )
        return None
    else:
        return imagesize.get(img_path)


def get_image_filesize(img_path):
    """
    Return the filesize of an image in bytes.
    """
    return pathlib.Path(img_path).stat().st_size


def get_image_hash(img_path):
    """
    Return the md5 hash of an image.
    """
    return hashlib.md5(img_path.read_bytes()).hexdigest()


def get_image_timestamp_from_filename(img_path) -> datetime.datetime:
    """
    def get_image_dimensions(img_path) -> tuple[int, int] | tuple[None, None] its filename.

        The timestamp must be in the format `YYYYMMDDHHMMSS` but can be
        preceded or followed by other characters (e.g. `84-20220916202959-snapshot.jpg`).

        >>> out_fmt = "%Y-%m-%d %H:%M:%S"
        >>> # Aarhus date format
        >>> get_image_timestamp_from_filename("20220810231507-00-07.jpg").strftime(out_fmt)
        '2022-08-10 23:15:07'
        >>> # Diopsis date format
        >>> get_image_timestamp_from_filename("20230124191342.jpg").strftime(out_fmt)
        '2023-01-24 19:13:42'
        >>> # Snapshot date format in Vermont traps
        >>> get_image_timestamp_from_filename("20220622000459-108-snapshot.jpg").strftime(out_fmt)
        '2022-06-22 00:04:59'
        >>> # Snapshot date format in Cyprus traps
        >>> get_image_timestamp_from_filename("84-20220916202959-snapshot.jpg").strftime(out_fmt)
        '2022-09-16 20:29:59'

    """
    name = pathlib.Path(img_path).stem
    date = None

    # Extract date from a filename using regex in the format %Y%m%d%H%M%S
    matches = re.search(r"(\d{14})", name)
    if matches:
        date = datetime.datetime.strptime(matches.group(), "%Y%m%d%H%M%S")
    else:
        date = dateutil.parser.parse(
            name, fuzzy=False
        )  # Fuzzy will interpret "DSC_1974" as 1974-01-01
        raise
    if date:
        return date
    else:
        raise ValueError(f"Could not parse date from filename '{img_path}'")


def get_image_timestamp_from_exif(img_path):
    """
    Parse the date and time a photo was taken from its EXIF data.

    This ignores the TimeZoneOffset and creates a datetime that is
    timezone naive.
    """
    exif = get_exif(img_path)
    datestring = exif["DateTime"].replace(":", "-", 2)
    date = dateutil.parser.parse(datestring)
    return date


def get_image_timestamp(
    img_path, check_exif=True, assert_exists=True
) -> datetime.datetime:
    """
    Parse the date and time a photo was taken from its filename or EXIF data.

    Reading the exif data is slow, so only do it if necessary.
    It is set to True for backwards compatibility.

    >>> images = pathlib.Path(TEST_IMAGES_BASE_PATH)
    >>> # Use filename
    >>> get_image_timestamp(images / "cyprus/84-20220916202959-snapshot.jpg").strftime("%Y-%m-%d %H:%M:%S")
    '2022-09-16 20:29:59'
    >>> # Fallback to EXIF
    >>> get_image_timestamp(images / "DSLR/DSC_0390.JPG").strftime("%Y-%m-%d %H:%M:%S")
    '2022-07-19 14:28:16'
    """
    if assert_exists:
        assert pathlib.Path(img_path).exists(), f"Image file does not exist: {img_path}"
    try:
        date = get_image_timestamp_from_filename(img_path)
    except (ValueError, ParserError) as e:
        if check_exif:
            logger.debug(f"Could not parse date from filename: {e}. Trying EXIF.")
            try:
                date = get_image_timestamp_from_exif(img_path)
            except dateutil.parser.ParserError:
                logger.error(
                    f"Could not parse image timestamp from filename or EXIF tags: {e}."
                )
                raise
        else:
            raise
    return date


def get_image_timestamp_with_timezone(img_path, default_offset="+0"):
    """
    Parse the date and time a photo was taken from its EXIF data.

    Also sets the timezone based on the TimeZoneOffset field if available.
    Example EXIF offset: "-4". Some libaries expect the format to be: "-04:00"
    However dateutil.parse seems to handle "-4" or "+4" just fine.
    """
    exif = get_exif(img_path)
    datestring = exif["DateTime"].replace(":", "-", 2)
    offset = exif.get("TimeZoneOffset") or str(default_offset)
    if int(offset) > 0:
        offset = f"+{offset}"
    datestring = f"{datestring} {offset}"
    date = dateutil.parser.parse(datestring)
    return date


def find_images(
    base_directory,
    absolute_paths=False,
    check_exif=True,
    skip_missing_timestamps=True,
    public_base_url=IMAGE_BASE_URL,
):
    logger.info(f"Scanning '{base_directory}' for images")
    base_directory = pathlib.Path(base_directory).expanduser().resolve()
    if not base_directory.exists():
        raise Exception(f"Directory does not exist: {base_directory}")
    extensions_list = "|".join([f.lstrip(".") for f in SUPPORTED_IMAGE_EXTENSIONS])
    pattern = rf"\.({extensions_list})$"
    for walk_path, _dirs, files in os.walk(base_directory):
        for name in files:
            if re.search(pattern, name, re.IGNORECASE):
                relative_path = pathlib.Path(walk_path) / name
                try:
                    relative_path = relative_path.relative_to(base_directory)
                except ValueError:
                    pass
                full_path = base_directory / relative_path
                path = full_path if absolute_paths else relative_path
                shape = get_image_dimensions(full_path)
                filesize = get_image_filesize(full_path)
                hash = get_image_hash(full_path)
                url = public_url(
                    relative_path, relative_to=base_directory, url_base=public_base_url
                )

                try:
                    date = get_image_timestamp(full_path, check_exif=check_exif)
                except Exception as e:
                    logger.error(
                        f"Skipping image, could not determine timestamp for: {full_path}\n {e}"
                    )
                    if skip_missing_timestamps:
                        continue
                    else:
                        date = None

                yield {
                    "path": str(path),
                    "timestamp": date,
                    "width": shape[0] if shape else None,
                    "height": shape[1] if shape else None,
                    "filesize": filesize,
                    "hash": hash,
                    "url": url,
                }


def group_images_by_day(images, maximum_gap_minutes=6 * 60):
    """
    Find consecutive images and group them into daily/nightly monitoring sessions.
    If the time between two photos is greater than `maximum_time_gap` (in minutes)
    then start a new session group. Each new group uses the first photo's day
    as the day of the session even if consecutive images are taken past midnight.
    # @TODO add other group by methods? like image size, camera model, random sample batches, etc. Add to UI settings

    @TODO make fake images for this test
    >>> images = find_images(TEST_IMAGES_BASE_PATH, skip_missing_timestamps=True)
    >>> sessions = group_images_by_day(images)
    >>> len(sessions)
    7
    """
    logger.info(
        f"Grouping images into date-based groups with a maximum gap of {maximum_gap_minutes} minutes"
    )
    images = sorted(images, key=lambda image: image["timestamp"])
    if not images:
        return {}

    groups = collections.OrderedDict()

    last_timestamp = None
    current_day = None

    for image in images:
        if last_timestamp:
            delta = (image["timestamp"] - last_timestamp).seconds / 60
        else:
            delta = maximum_gap_minutes

        logger.debug(f"{image['timestamp']}, {round(delta, 2)}")

        if delta >= maximum_gap_minutes:
            current_day = image["timestamp"].date()
            logger.debug(
                f"Gap of {round(delta/60, 1)} hours detected. Starting new session for date: {current_day}"
            )
            groups[current_day] = []

        groups[current_day].append(image)
        last_timestamp = image["timestamp"]

    # This is for debugging
    for day, images in groups.items():
        first_date = images[0]["timestamp"]
        last_date = images[-1]["timestamp"]
        delta = last_date - first_date
        hours = round(delta.seconds / 60 / 60, 1)
        logger.debug(
            f"Found session on {day} with {len(images)} images that ran for {hours} hours.\n"
            f"From {first_date.strftime('%c')} to {last_date.strftime('%c')}."
        )

    return groups
    # yield relative_path, get_image_timestamp(full_path)


def get_platform() -> Literal["win", "macosx", "linux", "unknown"]:
    """
    A string identifying the current operating system. It is one
    of: `'win'`, `'linux'`,  `'macosx'`, or `'unknown'`.
    Adapted from kivy.utils.platform()
    https://github.com/kivy/kivy/blob/master/kivy/utils.py
    """
    if _sys_platform in ("win32", "cygwin"):
        return "win"
    elif _sys_platform == "darwin":
        return "macosx"
    elif _sys_platform.startswith("linux"):
        return "linux"
    elif _sys_platform.startswith("freebsd"):
        return "linux"
    return "unknown"


def public_url(
    local_path: FilePath,
    relative_to: FilePath = None,
    url_base: str = IMAGE_BASE_URL,
) -> str:
    """
    Given a local path to a file, return a URL to that file.
    """

    path = pathlib.Path(local_path)
    try:
        path = path.relative_to(relative_to)
    except ValueError:
        pass
    path = path.as_posix()
    url = urllib.parse.urljoin(url_base, str(path).lstrip("/"))

    return url


def write_csv(
    images: Generator[dict, None, None],
    output_path: FilePath = None,
) -> pathlib.Path:
    """
    Write a list of images to a CSV file.
    """

    if not output_path:
        output_path = (
            f"inventory-{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        )
    output_path = pathlib.Path(output_path)

    columns = [
        "path",
        "timestamp",
        "height",
        "width",
        "filesize",
        "hash",
        "url",
    ]

    import tqdm

    file_count = 0
    unique_dirs = set()
    with open(output_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        print(f"Writing inventory to {output_path}")
        for image in tqdm.tqdm(images, desc="Found images"):
            writer.writerow(image)
            file_count += 1
            unique_dirs.add(pathlib.Path(image["path"]).parent)
            if file_count % 1000 == 0:
                print(
                    f" Found {file_count} in {len(unique_dirs)} directories and counting..."
                )

    print(f"Found {file_count} images in total.")
    return output_path


# Use argparse to create command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find images in a directory and write an inventory to CSV."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory to search for images",
    )

    parser.add_argument(
        "--public-base-url",
        type=str,
        default=IMAGE_BASE_URL,
        help="Base URL for constructing public image URLs",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output CSV file. Defaults to 'inventory-{timestamp}.csv' in the current directory.",
    )

    args = parser.parse_args()

    images = find_images(
        args.directory,
        public_base_url=args.public_base_url,
        absolute_paths=False,
    )

    # sessions = group_images_by_day(images, args.maximum_gap_minutes)
    # for day, images in sessions.items():
    #     print(f"Found session on {day} with {len(images)} images")

    write_csv(images, args.output)
