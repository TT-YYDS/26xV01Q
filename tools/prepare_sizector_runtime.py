"""Extract the supplied VC++ 2012 x64 runtime locally; never run its installer."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def prepare(sdk_root, destination):
    installer = Path(sdk_root) / "05_Environment" / "Microsoft_Visual_C++_Redistributable_2012_Update_4_x64.exe"
    data = installer.read_bytes()
    with TemporaryDirectory(prefix="sizector_runtime_") as temporary:
        temp = Path(temporary)
        offset, cabinets = 0, []
        while (offset := data.find(b"MSCF", offset)) >= 0:
            size = struct.unpack_from("<I", data, offset + 8)[0]
            if 36 <= size <= len(data) - offset:
                cabinet = temp / f"cabinet_{len(cabinets)}.cab"
                cabinet.write_bytes(data[offset:offset + size])
                folder = temp / cabinet.stem
                folder.mkdir()
                subprocess.run(["expand.exe", "-F:*", str(cabinet), str(folder)], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                cabinets.append(folder)
            offset += 4
        if not cabinets:
            raise RuntimeError("No embedded cabinets found")
        # Burn stores CAB payloads under IDs; the manifest maps these back to archives.
        payloads = {}
        for folder in cabinets:
            for file in folder.iterdir():
                try:
                    tree = ET.parse(file)
                except (ET.ParseError, UnicodeDecodeError):
                    continue
                for element in tree.iter():
                    filename = element.attrib.get("FilePath", "")
                    source = element.attrib.get("SourcePath", "")
                    if filename.lower().endswith(".cab") and source == Path(source).name:
                        payloads[source] = filename
        inner_folders = []
        for source_id in payloads:
            source = next((folder / source_id for folder in cabinets if (folder / source_id).is_file()), None)
            if source is None:
                raise RuntimeError(f"CAB payload not found: {source_id}")
            folder = temp / ("runtime_" + source_id)
            folder.mkdir()
            subprocess.run(["expand.exe", "-F:*", str(source), str(folder)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            inner_folders.append(folder)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("msvcr110.dll", "msvcp110.dll", "vcomp110.dll"):
            file_id = "F_CENTRAL_" + name.removesuffix(".dll") + "_x64"
            source = next((folder / file_id for folder in inner_folders if (folder / file_id).is_file()), None)
            if source is None:
                raise RuntimeError(f"Runtime payload not found: {name}")
            binary = source.read_bytes()
            pe = struct.unpack_from("<I", binary, 60)[0]
            if binary[:2] != b"MZ" or struct.unpack_from("<H", binary, pe + 4)[0] != 0x8664:
                raise RuntimeError(f"Not an x64 runtime: {name}")
            target = destination / name
            if target.exists() and target.read_bytes() != binary:
                raise RuntimeError(f"Existing runtime differs, refusing to replace: {target}")
            if not target.exists():
                target.write_bytes(binary)
        return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, default=ROOT / "盛相SDK")
    parser.add_argument("--destination", type=Path, default=ROOT / "settings" / "sizector_runtime" / "x64")
    args = parser.parse_args()
    print(prepare(args.sdk_root, args.destination))
