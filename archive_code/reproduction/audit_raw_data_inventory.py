"""Read official raw-data metadata and ZIP directories with bounded HTTP ranges."""

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import requests

SESSION = requests.Session()

ZENODO = [("XJTU", "10963339"), ("TJU", "6405084")]
MEND_FILES = "https://data.mendeley.com/public-api/datasets/nsc7hnsg4s/files?folder_id=root&version=2&%24start=0&%24limit=1000"
MIT_NAMES = [date + "_batchdata_updated_struct_errorcorrect.mat"
             for date in ["2017-05-12", "2017-06-30", "2018-04-12"]]


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, limit=4 * 1024 * 1024):
        self.url, self.size, self.limit, self.position = url, size, limit, 0
        self.downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size) + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Seek outside remote ZIP")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        if size > self.limit:
            raise ValueError("Requested remote read exceeds bounded limit")
        start = self.position
        with SESSION.get(self.url, headers={"Range": f"bytes={start}-{start + size - 1}"},
                          timeout=45, stream=True) as response:
            if response.status_code != 206:
                raise ValueError(f"Server did not honor bounded Range: HTTP {response.status_code}")
            content = response.raw.read(size + 1)
            if len(content) != size:
                raise ValueError("Unexpected ranged response length")
        self.position += size
        self.downloaded += size
        return content


def get_json(url, **kwargs):
    response = SESSION.get(url, timeout=45, **kwargs)
    response.raise_for_status()
    return response.json()


def inventory(output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, record in ZENODO:
        url = f"https://zenodo.org/api/records/{record}"
        metadata = get_json(url)
        (output / (dataset + "_official_metadata.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for file in metadata["files"]:
            if dataset == "TJU" and not file["key"].startswith("Dataset_"):
                continue
            records.append(dict(dataset=dataset, name=file["key"], compressed_bytes=file["size"],
                                checksum=file["checksum"], download_url=file["links"]["self"], metadata_url=url))
    hust = get_json(MEND_FILES, headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"})
    (output / "HUST_official_metadata.json").write_text(json.dumps(hust, indent=2), encoding="utf-8")
    for file in hust:
        details = file["content_details"]
        records.append(dict(dataset="HUST", name=file["filename"], compressed_bytes=details["size"],
                            checksum="sha256:" + details["sha256_hash"], download_url=details["download_url"], metadata_url=MEND_FILES))
    for record in records:
        try:
            remote = RemoteZip(record["download_url"], record["compressed_bytes"])
            with zipfile.ZipFile(remote) as archive:
                members = [dict(name=item.filename, compressed_bytes=item.compress_size,
                                uncompressed_bytes=item.file_size, compression=item.compress_type,
                                crc32=f"{item.CRC:08x}", header_offset=item.header_offset)
                           for item in archive.infolist()]
            record.update(archive_members=members,
                          uncompressed_bytes=sum(item["uncompressed_bytes"] for item in members),
                          member_count=len(members), metadata_range_downloaded_bytes=remote.downloaded)
        except (ValueError, requests.RequestException, zipfile.BadZipFile) as error:
            record.update(uncompressed_bytes=None, directory_error=str(error))
        print(record["dataset"], record["name"], record["compressed_bytes"], record.get("uncompressed_bytes"), flush=True)
        (output / "official_raw_inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    mit = []
    for name in MIT_NAMES:
        url = "https://data.matr.io/1/data/" + name
        response = SESSION.head(url, timeout=45, allow_redirects=True)
        mit.append(dict(dataset="MIT", name=name, url=url, http_status=response.status_code,
                        bytes=int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None,
                        content_type=response.headers.get("Content-Type"), final_url=response.url,
                        etag=response.headers.get("ETag"), accept_ranges=response.headers.get("Accept-Ranges")))
    (output / "MIT_raw_headers.json").write_text(json.dumps(mit, indent=2), encoding="utf-8")
    print(json.dumps(mit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/full_reproduction/feature_audit"))
    parser.add_argument("--direct", action="store_true", help="Use direct HTTPS without environment proxy/CA overrides")
    options = parser.parse_args()
    if options.direct:
        SESSION.trust_env = False
    inventory(options.output.resolve())
