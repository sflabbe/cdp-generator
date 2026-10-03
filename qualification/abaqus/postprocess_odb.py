"""ABAQUS-Q1 ODB extractor — run ONLY through the Abaqus Python interpreter.

Usage (launched by scripts/qualify_abaqus.py from the job directory)::

    abaqus python postprocess_odb.py <job>.odb <job>.odb_extract.json

It imports the proprietary ``odbAccess`` module and therefore must never be imported by the
ordinary package or test suite. It writes a small deterministic JSON document that normal
Python reads (schema ``abaqus_odb_extract.v1``). Variables that are not present in the ODB are
written as ``null`` and omitted from ``available_variables`` — absence is reported truthfully.

Kept compatible with both the Python 2.7 and Python 3 Abaqus interpreters.
"""

import json
import math
import sys

from abaqusConstants import INTEGRATION_POINT  # Abaqus-only
from odbAccess import openOdb  # Abaqus-only

SCHEMA_VERSION = "abaqus_odb_extract.v1"
SCALARS = ("PEEQ", "PEEQT", "DAMAGEC", "DAMAGET")
COMPONENT_11 = (("S", "S11"), ("E", "E11"), ("PE", "PE11"))


def _clean(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _find_set(odb, kind, name):
    for inst_name in sorted(odb.rootAssembly.instances.keys()):
        sets = getattr(odb.rootAssembly.instances[inst_name], kind)
        if name in sets.keys():
            return sets[name]
    sets = getattr(odb.rootAssembly, kind)
    if name in sets.keys():
        return sets[name]
    return None


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _element_values(field, region):
    try:
        subset = field.getSubset(region=region, position=INTEGRATION_POINT)
    except Exception:  # pragma: no cover - Abaqus-only
        subset = field.getSubset(region=region)
    return subset.values


def extract(odb_path, out_path):
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step_names = list(odb.steps.keys())
        step = odb.steps[step_names[-1]]
        eall = _find_set(odb, "elementSets", "EALL")
        x1 = _find_set(odb, "nodeSets", "X1")
        ref = _find_set(odb, "nodeSets", "REFNODE")
        available = set()
        frames = []
        for frame in step.frames:
            fields = frame.fieldOutputs
            record = {"step_time": _clean(frame.frameValue)}
            for key, label in COMPONENT_11:
                record[label] = None
                if key in fields.keys() and eall is not None:
                    vals = _element_values(fields[key], eall)
                    record[label] = _clean(_mean([v.data[0] for v in vals]))
                    if record[label] is not None:
                        available.add(label)
            for key in SCALARS:
                record[key] = None
                if key in fields.keys() and eall is not None:
                    vals = _element_values(fields[key], eall)
                    record[key] = _clean(_mean([v.data for v in vals]))
                    if record[key] is not None:
                        available.add(key)
            record["RF1_X1"] = None
            if "RF" in fields.keys() and x1 is not None:
                vals = fields["RF"].getSubset(region=x1).values
                record["RF1_X1"] = _clean(sum(v.data[0] for v in vals))
                if record["RF1_X1"] is not None:
                    available.add("RF1_X1")
            record["U1_REFNODE"] = None
            if "U" in fields.keys() and ref is not None:
                vals = fields["U"].getSubset(region=ref).values
                if vals:
                    record["U1_REFNODE"] = _clean(vals[0].data[0])
                    if record["U1_REFNODE"] is not None:
                        available.add("U1_REFNODE")
            frames.append(record)
        document = {
            "schema_version": SCHEMA_VERSION,
            "step": step_names[-1],
            "frame_count": len(frames),
            "available_variables": sorted(available),
            "frames": frames,
        }
    finally:
        odb.close()
    handle = open(out_path, "w")
    try:
        json.dump(document, handle, sort_keys=True, indent=1)
    finally:
        handle.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("usage: abaqus python postprocess_odb.py <job.odb> <out.json>\n")
        sys.exit(2)
    extract(sys.argv[1], sys.argv[2])
