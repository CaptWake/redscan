from typing import Any

from thrember.features import PEFeatureExtractor


def _feature_labels(feature: Any) -> list[str]:
    """Build labels in the exact order emitted by an EMBER2024 feature class."""

    name = feature.name
    labels: list[str]

    if name == "general":
        labels = [
            "size",
            "entropy",
            "is_pe",
            "start_byte_0",
            "start_byte_1",
            "start_byte_2",
            "start_byte_3",
        ]
    elif name == "histogram":
        labels = [f"byte_0x{value:02x}_frequency" for value in range(256)]
    elif name == "byteentropy":
        labels = [
            (
                f"entropy_{entropy / 2:.1f}-{(entropy + 1) / 2:.1f}."
                f"byte_0x{byte * 16:02x}-0x{byte * 16 + 15:02x}"
            )
            for entropy in range(16)
            for byte in range(16)
        ]
    elif name == "strings":
        regex_names = [
            regex
            for regex, _ in sorted(feature.regex_idxs.items(), key=lambda item: item[1])
        ]
        labels = ["count", "average_length", "printable_count"]
        labels.extend(
            f"printable_0x{value:02x}_frequency" for value in range(0x20, 0x80)
        )
        labels.append("entropy")
        labels.extend(f"pattern.{regex}" for regex in regex_names)
    elif name == "header":
        labels = [
            "coff.timestamp",
            "coff.number_of_sections",
            "coff.number_of_symbols",
            "coff.sizeof_optional_header",
            "coff.pointer_to_symbol_table",
            "coff.machine",
            "optional.subsystem",
            "optional.major_image_version",
            "optional.minor_image_version",
            "optional.major_linker_version",
            "optional.minor_linker_version",
            "optional.major_operating_system_version",
            "optional.minor_operating_system_version",
            "optional.major_subsystem_version",
            "optional.minor_subsystem_version",
            "optional.sizeof_code",
            "optional.sizeof_headers",
            "optional.sizeof_image",
            "optional.sizeof_initialized_data",
            "optional.sizeof_uninitialized_data",
            "optional.sizeof_stack_reserve",
            "optional.sizeof_stack_commit",
            "optional.sizeof_heap_reserve",
            "optional.sizeof_heap_commit",
            "optional.address_of_entrypoint",
            "optional.base_of_code",
            "optional.image_base",
            "optional.section_alignment",
            "optional.checksum",
            "optional.number_of_rvas_and_sizes",
        ]
        labels.extend(
            f"coff.characteristic.{item}" for item in feature._image_characteristics
        )
        labels.extend(
            f"optional.dll_characteristic.{item}"
            for item in feature._dll_characteristics
        )
        labels.extend(f"dos.{item}" for item in feature._dos_members)
    elif name == "section":
        labels = [
            "count",
            "zero_size_count",
            "empty_name_count",
            "read_execute_count",
            "writable_count",
            "max_entropy",
            "min_entropy",
            "max_size_ratio",
            "min_size_ratio",
            "max_virtual_size_ratio",
            "min_virtual_size_ratio",
        ]
        for family in ("size", "virtual_size", "entropy", "characteristics"):
            labels.extend(f"hashed_{family}_bucket_{index}" for index in range(50))
        labels.extend(f"hashed_entry_name_bucket_{index}" for index in range(10))
        labels.extend(("overlay.size", "overlay.size_ratio", "overlay.entropy"))
    elif name == "imports":
        labels = ["function_count", "library_count"]
        labels.extend(f"hashed_library_bucket_{index}" for index in range(256))
        labels.extend(f"hashed_function_bucket_{index}" for index in range(1024))
    elif name == "exports":
        labels = ["count"]
        labels.extend(f"hashed_name_bucket_{index}" for index in range(128))
    elif name == "datadirectories":
        labels = []
        for directory in feature._name_order:
            labels.extend(
                (
                    f"{directory.lower()}.size",
                    f"{directory.lower()}.virtual_address",
                )
            )
        labels.extend(("has_relocations", "has_dynamic_relocations"))
    elif name == "richheader":
        labels = ["pair_count"]
        labels.extend(f"hashed_pair_bucket_{index}" for index in range(32))
    elif name == "authenticode":
        labels = [
            "certificate_count",
            "self_signed",
            "empty_program_name",
            "no_countersigner",
            "parse_error",
            "chain_max_depth",
            "latest_signing_time",
            "signing_time_difference",
        ]
    elif name == "pefilewarnings":
        warning_names = {
            index: warning for warning, index in feature.warning_ids.items()
        }
        labels = [
            warning_names.get(index, f"warning_{index}")
            for index in range(feature.dim - 1)
        ]
        labels.append("count")
    else:
        labels = [f"feature_{index}" for index in range(feature.dim)]

    if len(labels) != feature.dim:
        return [f"{name}.feature_{index}" for index in range(feature.dim)]
    return [f"{name}.{label}" for label in labels]


def get_feature_metadata(
    extractor: PEFeatureExtractor,
) -> tuple[list[str], list[str]]:
    """Return labels and group names aligned with an extractor vector."""

    labels: list[str] = []
    groups: list[str] = []
    for feature in extractor.features:
        labels.extend(_feature_labels(feature))
        groups.extend([feature.name] * feature.dim)

    if len(labels) != extractor.dim or len(groups) != extractor.dim:
        raise ValueError("EMBER feature metadata does not match extractor output")
    return labels, groups
